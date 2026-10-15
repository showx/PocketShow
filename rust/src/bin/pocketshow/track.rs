use super::yolo::Det;

#[derive(Clone, Debug)]
pub struct Person {
    pub id: i64,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
    pub vx: f32,
    pub vy: f32,
    pub person_id: Option<String>,
    pub name: Option<String>,
    pub face: Option<[f32; 15]>,
    pub embedding: Option<Vec<f32>>,
    pub body: Option<Vec<f32>>,
}

pub struct Tracker {
    next_id: i64,
    prev: Vec<(i64, f32, f32, f32, f32)>,
    misses: Vec<(i64, u32)>,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            prev: Vec::new(),
            misses: Vec::new(),
        }
    }

    pub fn update(&mut self, dets: &[Det]) -> Vec<Person> {
        let mut pairs = Vec::new();
        for (di, det) in dets.iter().enumerate() {
            for (pid, x1, y1, x2, y2) in &self.prev {
                let iou = box_iou(det.x1, det.y1, det.x2, det.y2, *x1, *y1, *x2, *y2);
                if iou >= 0.3 {
                    pairs.push((iou, *pid, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut used_id = vec![false; self.prev.len()];
        let mut used_det = vec![false; dets.len()];
        let mut people = Vec::new();
        for (_, pid, di) in pairs {
            let Some(prev_index) = self.prev.iter().position(|item| item.0 == pid) else {
                continue;
            };
            if used_id[prev_index] || used_det[di] {
                continue;
            }
            used_id[prev_index] = true;
            used_det[di] = true;
            let det = &dets[di];
            let (_, x1, y1, x2, y2) = self.prev[prev_index];
            people.push(person_from(pid, det, (det.x1 + det.x2 - x1 - x2) * 0.5, (det.y1 + det.y2 - y1 - y2) * 0.5));
        }
        for (di, det) in dets.iter().enumerate() {
            if used_det[di] {
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            people.push(person_from(id, det, 0.0, 0.0));
        }
        let mut misses = Vec::new();
        for (index, (pid, x1, y1, x2, y2)) in self.prev.iter().enumerate() {
            if used_id[index] {
                continue;
            }
            let seen = self.misses.iter().find(|(id, _)| id == pid).map(|(_, n)| *n).unwrap_or(0) + 1;
            if seen <= 12 {
                misses.push((*pid, seen, *x1, *y1, *x2, *y2));
            }
        }
        self.prev = people
            .iter()
            .map(|person| (person.id, person.x1, person.y1, person.x2, person.y2))
            .chain(misses.iter().map(|item| (item.0, item.2, item.3, item.4, item.5)))
            .collect();
        self.misses = misses.into_iter().map(|item| (item.0, item.1)).collect();
        people
    }
}

fn person_from(id: i64, det: &Det, vx: f32, vy: f32) -> Person {
    Person {
        id,
        x1: det.x1,
        y1: det.y1,
        x2: det.x2,
        y2: det.y2,
        conf: det.conf,
        vx,
        vy,
        person_id: None,
        name: None,
        face: None,
        embedding: None,
        body: None,
    }
}

fn box_iou(ax1: f32, ay1: f32, ax2: f32, ay2: f32, bx1: f32, by1: f32, bx2: f32, by2: f32) -> f32 {
    let inter = (ax2.min(bx2) - ax1.max(bx1)).max(0.0) * (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_box_keeps_id() {
        let mut tracker = Tracker::new();
        let first = tracker.update(&[Det { x1: 10.0, y1: 10.0, x2: 50.0, y2: 80.0, conf: 0.9 }]);
        let second = tracker.update(&[Det { x1: 12.0, y1: 11.0, x2: 52.0, y2: 82.0, conf: 0.9 }]);
        assert_eq!(first[0].id, second[0].id);
        assert!(second[0].vx > 0.0);
    }
}
