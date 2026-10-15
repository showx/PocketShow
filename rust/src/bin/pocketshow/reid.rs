use std::collections::HashMap;

use anyhow::Result;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use super::track::Person;

pub struct Body {
    session: Session,
    input_name: String,
    cache: HashMap<i64, (u64, Vec<f32>)>,
}

impl Body {
    pub fn load(env: &ort::environment::Environment, path: &std::path::Path) -> Result<Self> {
        let session = Session::builder(env)?.commit_from_file(path)?;
        let input_name = session.inputs()[0].name().to_string();
        Ok(Self {
            session,
            input_name,
            cache: HashMap::new(),
        })
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn annotate(&mut self, rgb: &[u8], width: u32, height: u32, people: &mut [Person], frame: u64) -> Result<()> {
        let live: Vec<i64> = people.iter().map(|person| person.id).collect();
        self.cache.retain(|id, _| live.contains(id));
        let mut fresh = 0;
        for person in people.iter_mut() {
            if let Some((at, embedding)) = self.cache.get(&person.id) {
                if frame.saturating_sub(*at) < 15 {
                    person.body = Some(embedding.clone());
                    continue;
                }
            }
            if fresh >= 4 || person.y2 - person.y1 < 24.0 {
                continue;
            }
            fresh += 1;
            if let Some(embedding) = self.embed(rgb, width, height, person)? {
                self.cache.insert(person.id, (frame, embedding.clone()));
                person.body = Some(embedding);
            }
        }
        Ok(())
    }

    fn embed(&mut self, rgb: &[u8], width: u32, height: u32, person: &Person) -> Result<Option<Vec<f32>>> {
        let bw = (person.x2 - person.x1).max(1.0);
        let bh = (person.y2 - person.y1).max(1.0);
        let x1 = (person.x1 - bw * 0.08).clamp(0.0, width as f32 - 1.0) as u32;
        let y1 = (person.y1 - bh * 0.08).clamp(0.0, height as f32 - 1.0) as u32;
        let x2 = (person.x2 + bw * 0.08).clamp(0.0, width as f32).max(x1 as f32 + 1.0) as u32;
        let y2 = (person.y2 + bh * 0.08).clamp(0.0, height as f32).max(y1 as f32 + 1.0) as u32;
        let crop_w = x2 - x1;
        let crop_h = y2 - y1;
        let mut tensor = vec![0.0f32; 3 * 256 * 128];
        for y in 0..256usize {
            let src_y = (y as f32 + 0.5) * crop_h as f32 / 256.0 - 0.5;
            let (y0, y1_src, fy) = bilinear_index(src_y, crop_h);
            for x in 0..128usize {
                let src_x = (x as f32 + 0.5) * crop_w as f32 / 128.0 - 0.5;
                let (x0, x1_src, fx) = bilinear_index(src_x, crop_w);
                let sample = |sx: u32, sy: u32, channel: usize| {
                    let pixel = (((y1 + sy) * width + (x1 + sx)) * 3) as usize + channel;
                    if pixel < rgb.len() { rgb[pixel] as f32 } else { 0.0 }
                };
                for channel in 0..3 {
                    let src_channel = 2 - channel;
                    let value = sample(x0, y0, src_channel) * (1.0 - fx) * (1.0 - fy)
                        + sample(x0, y1_src, src_channel) * (1.0 - fx) * fy
                        + sample(x1_src, y0, src_channel) * fx * (1.0 - fy)
                        + sample(x1_src, y1_src, src_channel) * fx * fy;
                    tensor[(channel * 256 + y) * 128 + x] = value.round().clamp(0.0, 255.0);
                }
            }
        }
        let array = Array4::from_shape_vec((1, 3, 256, 128), tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => TensorRef::from_array_view(&array)?])?;
        let mut values: Vec<f32> = outputs[0].try_extract_array::<f32>()?.iter().copied().collect();
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm < 1e-6 {
            return Ok(None);
        }
        for value in &mut values {
            *value /= norm;
        }
        Ok(Some(values))
    }

    pub fn seed(&mut self, gallery: &mut super::gallery::Gallery) -> Result<usize> {
        let covers = gallery.covers();
        let mut filled = 0;
        for (id, path) in covers {
            let image = image::open(&path).ok();
            let Some(image) = image else { continue };
            let rgb = image.to_rgb8();
            let (width, height) = rgb.dimensions();
            if width < 8 || height < 8 {
                continue;
            }
            let person = Person {
                id: 0,
                x1: 0.0,
                y1: 0.0,
                x2: width as f32,
                y2: height as f32,
                conf: 1.0,
                vx: 0.0,
                vy: 0.0,
                person_id: None,
                name: None,
                face: None,
                embedding: None,
                body: None,
            };
            if let Some(embedding) = self.embed(rgb.as_raw(), width, height, &person)? {
                gallery.replace_body(&id, embedding);
                filled += 1;
            }
        }
        Ok(filled)
    }
}

fn bilinear_index(src: f32, length: u32) -> (u32, u32, f32) {
    if length <= 1 {
        return (0, 0, 0.0);
    }
    let max = length - 1;
    if src <= 0.0 {
        return (0, 0, 0.0);
    }
    let base = src.floor();
    let mut index = base as u32;
    let mut frac = src - base;
    if index >= max {
        index = max;
        frac = 0.0;
    }
    let next = (index + 1).min(max);
    if next == index {
        frac = 0.0;
    }
    (index, next, frac.clamp(0.0, 1.0))
}
