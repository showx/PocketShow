use std::fs;
use std::path::{Path, PathBuf};
use anyhow::Result;
use serde_json::{json, Value};

use super::track::Person;

#[derive(Clone, Debug)]
struct Seat {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
}

#[derive(Clone, Debug)]
struct Record {
    id: String,
    name: String,
    faces: Vec<Vec<f32>>,
    bodies: Vec<Vec<f32>>,
    seats: Vec<(String, Seat)>,
    photo: Option<String>,
}

pub struct Gallery {
    path: PathBuf,
    people: Vec<Record>,
    document: Option<Value>,
}

impl Gallery {
    pub fn load(path: &Path) -> Result<Self> {
        let path = roster_path(path);
        if !path.is_file() {
            return Ok(Self {
                path,
                people: Vec::new(),
                document: None,
            });
        }
        let document: Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
        let people = document
            .get("people")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(parse_record).collect())
            .unwrap_or_default();
        Ok(Self {
            path,
            people,
            document: Some(document),
        })
    }

    pub fn len(&self) -> usize {
        self.people.len()
    }

    pub fn covers(&self) -> Vec<(String, PathBuf)> {
        let root = self.path.parent().unwrap_or(Path::new(".")).join("faces");
        self.people
            .iter()
            .filter_map(|record| {
                let rel = record.photo.clone().unwrap_or_else(|| format!("{}/cover.jpg", record.id));
                let path = root.join(rel);
                path.is_file().then(|| (record.id.clone(), path))
            })
            .collect()
    }

    pub fn replace_body(&mut self, id: &str, embedding: Vec<f32>) {
        if let Some(record) = self.people.iter_mut().find(|record| record.id == id) {
            record.bodies = vec![embedding];
        }
    }

    pub fn search_windows(&self, camera: &str) -> Vec<(f32, f32, f32, f32)> {
        let mut windows = Vec::new();
        for record in &self.people {
            for (id, seat) in &record.seats {
                if id != camera {
                    continue;
                }
                let half_w = (seat.rx * 1.2).max(0.05);
                let tight = (
                    (seat.cx - half_w).clamp(0.0, 1.0),
                    (seat.cy - (seat.ry * 1.6).max(0.16)).clamp(0.0, 1.0),
                    (seat.cx + half_w).clamp(0.0, 1.0),
                    (seat.cy + (seat.ry * 0.8).max(0.08)).clamp(0.0, 1.0),
                );
                let tall = (
                    (seat.cx - 0.07).clamp(0.0, 1.0),
                    (seat.cy - 0.22).clamp(0.0, 1.0),
                    (seat.cx + 0.07).clamp(0.0, 1.0),
                    (seat.cy + 0.10).clamp(0.0, 1.0),
                );
                windows.push(tight);
                if (tall.1 - tight.1).abs() >= 0.02 || (tall.0 - tight.0).abs() >= 0.02 {
                    windows.push(tall);
                }
            }
        }
        windows
    }

    pub fn seated(&self, camera: &str) -> Vec<(String, String)> {
        self.people
            .iter()
            .filter(|record| record.seats.iter().any(|(id, _)| id == camera))
            .map(|record| (record.id.clone(), record.name.clone()))
            .collect()
    }

    pub fn pin(
        &mut self,
        camera_id: &str,
        camera_name: &str,
        person_id: &str,
        name: &str,
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        face: Option<Vec<f32>>,
        body: Option<Vec<f32>>,
    ) -> Result<(String, String)> {
        if camera_id.is_empty() {
            return Err(anyhow::anyhow!("这路画面还没有摄像机编号，不能记工位"));
        }
        let cx = cx.clamp(0.0, 1.0);
        let cy = cy.clamp(0.0, 1.0);
        let rx = rx.clamp(0.025, 0.16);
        let ry = ry.clamp(0.05, 0.28);
        self.ensure_document();
        let id = if person_id.is_empty() {
            let name = name.trim();
            if name.is_empty() {
                return Err(anyhow::anyhow!("先填写姓名"));
            }
            if let Some(existing) = self.people.iter().find(|record| record.name == name) {
                existing.id.clone()
            } else {
                let id = self.next_id();
                self.people.push(Record {
                    id: id.clone(),
                    name: name.to_string(),
                    faces: Vec::new(),
                    bodies: Vec::new(),
                    seats: Vec::new(),
                    photo: None,
                });
                if let Some(items) = self.people_json_mut() {
                    items.push(json!({
                        "id": id,
                        "name": name,
                        "note": "",
                        "guest": false,
                        "embedding": [],
                        "samples": 0,
                        "photo": "",
                        "templates": [],
                        "appearance": [],
                        "appearances": [],
                        "seats": {}
                    }));
                }
                id
            }
        } else {
            person_id.to_string()
        };
        if !self.people.iter().any(|record| record.id == id) {
            return Err(anyhow::anyhow!("人物库里没有这个人"));
        }
        self.vacate(&id, camera_id, cx, cy, rx, ry);
        if let Some(record) = self.people.iter_mut().find(|record| record.id == id) {
            record.seats.retain(|(seat_camera, _)| seat_camera != camera_id);
            record.seats.push((
                camera_id.to_string(),
                Seat { cx, cy, rx, ry },
            ));
            if let Some(face) = face.clone() {
                if !face.is_empty() {
                    record.faces.push(face);
                    record.faces.truncate(6);
                }
            }
            if let Some(body) = body.clone() {
                if !body.is_empty() {
                    record.bodies = vec![body];
                }
            }
        }
        let saved_name = self.people.iter().find(|record| record.id == id).map(|record| record.name.clone()).unwrap_or_default();
        if let Some(item) = self.find_json_mut(&id) {
            let seats = item.as_object_mut().and_then(|obj| {
                obj.entry("seats").or_insert_with(|| json!({}));
                obj.get_mut("seats").and_then(Value::as_object_mut)
            });
            if let Some(seats) = seats {
                seats.insert(
                    camera_id.to_string(),
                    json!({
                        "cx": cx,
                        "cy": cy,
                        "rx": rx,
                        "ry": ry,
                        "hits": 999,
                        "locked": true,
                        "camera_name": camera_name
                    }),
                );
            }
            if let Some(face) = &face {
                if !face.is_empty() {
                    if let Some(templates) = item.get_mut("templates").and_then(Value::as_array_mut) {
                        templates.push(json!(face));
                        let extra = templates.len().saturating_sub(6);
                        templates.drain(0..extra);
                    }
                    if item.get("embedding").and_then(Value::as_array).is_some_and(Vec::is_empty) {
                        item["embedding"] = json!(face);
                    }
                }
            }
            if let Some(body) = &body {
                if !body.is_empty() {
                    item["appearance"] = json!(body);
                }
            }
        }
        self.save()?;
        Ok((id, saved_name))
    }

    pub fn save_cover(&mut self, id: &str, rgb: &[u8], width: u32, height: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> Result<()> {
        let x1 = x1.clamp(0.0, width.saturating_sub(1) as f32) as u32;
        let y1 = y1.clamp(0.0, height.saturating_sub(1) as f32) as u32;
        let x2 = x2.clamp(x1 as f32 + 1.0, width as f32) as u32;
        let y2 = y2.clamp(y1 as f32 + 1.0, height as f32) as u32;
        let crop_w = x2 - x1;
        let crop_h = y2 - y1;
        if crop_w < 8 || crop_h < 8 {
            return Ok(());
        }
        let mut crop = Vec::with_capacity((crop_w * crop_h * 3) as usize);
        for row in y1..y2 {
            let start = ((row * width + x1) * 3) as usize;
            let end = start + (crop_w * 3) as usize;
            if end <= rgb.len() {
                crop.extend_from_slice(&rgb[start..end]);
            }
        }
        let Some(image) = image::RgbImage::from_raw(crop_w, crop_h, crop) else {
            return Ok(());
        };
        let folder = self.path.parent().unwrap_or(Path::new(".")).join("faces").join(id);
        fs::create_dir_all(&folder)?;
        let path = folder.join("cover.jpg");
        image.save(&path)?;
        let rel = format!("{id}/cover.jpg");
        if let Some(record) = self.people.iter_mut().find(|record| record.id == id) {
            record.photo = Some(rel.clone());
        }
        if let Some(item) = self.find_json_mut(id) {
            item["photo"] = json!(rel);
            item["photo_area"] = json!(crop_w * crop_h);
        }
        self.save()?;
        Ok(())
    }

    fn next_id(&self) -> String {
        let max = self
            .people
            .iter()
            .filter_map(|record| record.id.strip_prefix('p').and_then(|rest| rest.parse::<u32>().ok()))
            .max()
            .unwrap_or(0);
        format!("p{:03}", max + 1)
    }

    fn ensure_document(&mut self) {
        if self.document.is_none() {
            self.document = Some(json!({"people": []}));
        }
    }

    fn people_json_mut(&mut self) -> Option<&mut Vec<Value>> {
        self.document.as_mut()?.get_mut("people")?.as_array_mut()
    }

    fn find_json_mut(&mut self, id: &str) -> Option<&mut Value> {
        self.people_json_mut()?.iter_mut().find(|item| item.get("id").and_then(Value::as_str) == Some(id))
    }

    fn vacate(&mut self, keep: &str, camera: &str, cx: f32, cy: f32, rx: f32, ry: f32) {
        let mut dropped = Vec::new();
        for record in &mut self.people {
            if record.id == keep {
                continue;
            }
            let overlap = record.seats.iter().any(|(seat_camera, seat)| {
                if seat_camera != camera {
                    return false;
                }
                let dx = (cx - seat.cx) / rx.max(seat.rx).max(1e-3);
                let dy = (cy - seat.cy) / ry.max(seat.ry).max(1e-3);
                (dx * dx + dy * dy).sqrt() < 1.2
            });
            if overlap {
                record.seats.retain(|(seat_camera, _)| seat_camera != camera);
                dropped.push(record.id.clone());
            }
        }
        for id in dropped {
            if let Some(item) = self.find_json_mut(&id) {
                if let Some(seats) = item.get_mut("seats").and_then(Value::as_object_mut) {
                    seats.remove(camera);
                }
            }
        }
    }

    pub fn cards(&self) -> Vec<(String, String, usize)> {
        self.people
            .iter()
            .map(|record| (record.id.clone(), record.name.clone(), record.faces.len().max(record.bodies.len())))
            .collect()
    }

    pub fn rename(&mut self, id: &str, name: &str) -> bool {
        let Some(record) = self.people.iter_mut().find(|record| record.id == id) else {
            return false;
        };
        let name = name.trim();
        if name.is_empty() {
            return false;
        }
        record.name = name.to_string();
        let id = record.id.clone();
        self.write_name(&id, name);
        let _ = self.save();
        true
    }

    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.people.len();
        self.people.retain(|record| record.id != id);
        if self.people.len() == before {
            return false;
        }
        let _ = self.save();
        true
    }

    pub fn apply(&mut self, people: &mut [Person], camera: &str, width: u32, height: u32, threshold: f32, margin: f32) {
        let width = width.max(1) as f32;
        let height = height.max(1) as f32;
        let seats = assign_seats(&self.people, people, camera, width, height);
        for (index, person) in people.iter_mut().enumerate() {
            let face = person.embedding.as_ref().and_then(|embedding| rank(&self.people, embedding, true));
            let body = person.body.as_ref().and_then(|embedding| rank(&self.people, embedding, false));
            let seat = seats.iter().find(|hit| hit.person == index).map(|hit| {
                let sim = person
                    .body
                    .as_ref()
                    .map(|embedding| best_cosine(embedding, &self.people[hit.record].bodies))
                    .unwrap_or(-1.0);
                (hit.record, sim, hit.dist)
            });
            if let Some((id, name)) = fuse(&self.people, face, body, seat, threshold, margin) {
                person.person_id = Some(id);
                person.name = Some(name);
            }
        }
    }

    fn write_name(&mut self, id: &str, name: &str) {
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let Some(items) = document.get_mut("people").and_then(Value::as_array_mut) else {
            return;
        };
        if let Some(item) = items.iter_mut().find(|item| item.get("id").and_then(Value::as_str) == Some(id)) {
            item["name"] = json!(name);
        }
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(document) = &self.document {
            fs::write(&self.path, serde_json::to_string_pretty(document)?)?;
        }
        Ok(())
    }
}

fn roster_path(path: &Path) -> PathBuf {
    if path.is_file() && path.file_name().and_then(|name| name.to_str()) == Some("faces.json") {
        return path.to_path_buf();
    }
    if let Some(parent) = path.parent() {
        let faces = parent.join("faces.json");
        if faces.is_file() {
            return faces;
        }
    }
    path.to_path_buf()
}

fn parse_record(value: &Value) -> Option<Record> {
    let id = value.get("id").and_then(Value::as_str)?.to_string();
    let name = value.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
    let templates = vector_list(value.get("templates"));
    let faces = if templates.is_empty() { vector_list(value.get("embedding")).into_iter().chain(vector_list(value.get("embeddings"))).collect() } else { templates };
    let appearances = vector_list(value.get("appearances"));
    let bodies = if appearances.is_empty() { vector_list(value.get("appearance")) } else { appearances };
    let mut seats = Vec::new();
    if let Some(map) = value.get("seats").and_then(Value::as_object) {
        for (camera, seat) in map {
            let cx = seat.get("cx").and_then(Value::as_f64)? as f32;
            let cy = seat.get("cy").and_then(Value::as_f64)? as f32;
            seats.push((
                camera.clone(),
                Seat {
                    cx,
                    cy,
                    rx: seat.get("rx").and_then(Value::as_f64).unwrap_or(0.05) as f32,
                    ry: seat.get("ry").and_then(Value::as_f64).unwrap_or(0.10) as f32,
                },
            ));
        }
    }
    let photo = value.get("photo").and_then(Value::as_str).map(str::to_string);
    Some(Record { id, name, faces, bodies, seats, photo })
}

fn vector_list(value: Option<&Value>) -> Vec<Vec<f32>> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };
    if items.first().and_then(Value::as_array).is_some() {
        return items.iter().filter_map(Value::as_array).map(|item| floats(item)).filter(|item| !item.is_empty()).collect();
    }
    let one = floats(items);
    if one.is_empty() { Vec::new() } else { vec![one] }
}

fn floats(items: &[Value]) -> Vec<f32> {
    items.iter().filter_map(Value::as_f64).map(|value| value as f32).collect()
}

struct SeatHit {
    person: usize,
    record: usize,
    dist: f32,
    body: f32,
}

fn assign_seats(records: &[Record], people: &[Person], camera: &str, width: f32, height: f32) -> Vec<SeatHit> {
    let mut hits = Vec::new();
    for (person_index, person) in people.iter().enumerate() {
        for (record_index, record) in records.iter().enumerate() {
            for (seat_camera, seat) in &record.seats {
                if seat_camera != camera {
                    continue;
                }
                let dist = box_seat_dist(seat, person, width, height);
                if dist <= 1.25 {
                    let body = person
                        .body
                        .as_ref()
                        .map(|embedding| best_cosine(embedding, &record.bodies))
                        .unwrap_or(-1.0);
                    hits.push(SeatHit { person: person_index, record: record_index, dist, body });
                }
            }
        }
    }
    hits.sort_by(|a, b| a.dist.total_cmp(&b.dist).then(b.body.total_cmp(&a.body)));
    let mut used_person = vec![false; people.len()];
    let mut used_record = vec![false; records.len()];
    let mut kept = Vec::new();
    for hit in hits {
        if used_person[hit.person] || used_record[hit.record] {
            continue;
        }
        used_person[hit.person] = true;
        used_record[hit.record] = true;
        kept.push(hit);
    }
    kept
}

fn box_seat_dist(seat: &Seat, person: &Person, width: f32, height: f32) -> f32 {
    let nx1 = (person.x1 / width).clamp(0.0, 1.0);
    let ny1 = (person.y1 / height).clamp(0.0, 1.0);
    let nx2 = (person.x2 / width).clamp(0.0, 1.0);
    let bottom = (person.y2 / height).clamp(0.0, 1.0);
    let box_h = (bottom - ny1).max(0.0);
    let ry = seat.ry.max(1e-4);
    let max_drop = 0.10_f32.max(box_h * 1.15 + 0.05).max(ry * 1.3);
    let mut reach = 0.045_f32.max(box_h * 0.5);
    if box_h <= 0.22 {
        reach = reach.max(max_drop.min(0.16));
    }
    let ny2 = (bottom + reach).clamp(0.0, 1.0);
    if seat.cy - bottom > max_drop {
        return 9.0;
    }
    let px = seat.cx.clamp(nx1.min(nx2), nx1.max(nx2));
    let py = seat.cy.clamp(ny1.min(ny2), ny1.max(ny2));
    let dx = (px - seat.cx) / seat.rx.max(1e-4);
    let dy = (py - seat.cy) / ry;
    (dx * dx + dy * dy).sqrt()
}

fn rank(records: &[Record], embedding: &[f32], face: bool) -> Option<(f32, f32, usize)> {
    let mut best = -1.0;
    let mut second = -1.0;
    let mut index = 0;
    let mut any = false;
    for (record_index, record) in records.iter().enumerate() {
        let stored = if face { &record.faces } else { &record.bodies };
        if stored.is_empty() {
            continue;
        }
        let sim = best_cosine(embedding, stored);
        if sim > best {
            second = best;
            best = sim;
            index = record_index;
            any = true;
        } else if sim > second {
            second = sim;
        }
    }
    any.then_some((best, second.max(0.0), index))
}

fn fuse(
    records: &[Record],
    face: Option<(f32, f32, usize)>,
    body: Option<(f32, f32, usize)>,
    seat: Option<(usize, f32, f32)>,
    threshold: f32,
    margin: f32,
) -> Option<(String, String)> {
    let name = |index: usize| (records[index].id.clone(), records[index].name.clone());
    let face_ok = face.filter(|(sim, second, _)| *sim >= threshold && *sim - *second >= margin);
    let body_ok = body.filter(|(sim, second, _)| *sim >= 0.40 && *sim - *second >= 0.08);
    let seat_index = seat.map(|(index, _, _)| index);
    let seat_body = seat.map(|(_, sim, _)| sim).unwrap_or(-1.0);
    let seat_dist = seat.map(|(_, _, dist)| dist).unwrap_or(9.0);
    if let Some(seat_index) = seat_index {
        let rival = body.is_some_and(|(sim, second, index)| index != seat_index && sim >= 0.42 && sim - second >= 0.08);
        let body_agrees = seat_body >= 0.32;
        let face_agrees = face.is_some_and(|(sim, _, index)| index == seat_index && sim >= 0.40);
        if !rival && (seat_dist <= 0.85 || ((body_agrees || face_agrees) && seat_dist <= 1.15)) {
            return Some(name(seat_index));
        }
    }
    if let Some((_, _, index)) = face_ok {
        if let Some(seat_index) = seat_index {
            if index != seat_index {
                if body_ok.is_some_and(|(_, _, body_index)| body_index == index) {
                    return Some(name(index));
                }
                if seat_body >= 0.38 {
                    return Some(name(seat_index));
                }
                let face_sim = face.map(|(sim, _, _)| sim).unwrap_or(0.0);
                if face_sim - seat_body < margin {
                    if seat_body >= 0.42 {
                        return Some(name(seat_index));
                    }
                    return None;
                }
            }
        }
        return Some(name(index));
    }
    if let (Some((sim, _, index)), Some(seat_index)) = (face, seat_index) {
        if index == seat_index && sim >= 0.42 {
            return Some(name(index));
        }
    }
    if let (Some((sim, _, index)), Some(seat_index)) = (body, seat_index) {
        if index == seat_index && sim >= 0.38 {
            return Some(name(seat_index));
        }
    }
    if let Some(seat_index) = seat_index {
        if seat_body >= 0.38 {
            return Some(name(seat_index));
        }
        return None;
    }
    body_ok.map(|(_, _, index)| name(index))
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (left, right) in a.iter().zip(b.iter()) {
        dot += left * right;
        na += left * left;
        nb += right * right;
    }
    if na < 1e-6 || nb < 1e-6 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn best_cosine(embedding: &[f32], stored: &[Vec<f32>]) -> f32 {
    stored.iter().map(|item| cosine(embedding, item)).fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_same_direction() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-5);
    }
}
