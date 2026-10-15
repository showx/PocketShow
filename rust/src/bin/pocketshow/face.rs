use anyhow::Result;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use super::track::Person;

const REF: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

pub struct Faces {
    detect: Session,
    embed: Session,
}

impl Faces {
    pub fn load(env: &ort::environment::Environment, yunet: &std::path::Path, sface: &std::path::Path) -> Result<Self> {
        Ok(Self {
            detect: Session::builder(env)?.commit_from_file(yunet)?,
            embed: Session::builder(env)?.commit_from_file(sface)?,
        })
    }

    pub fn annotate(&mut self, rgb: &[u8], width: u32, height: u32, people: &mut [Person]) -> Result<()> {
        let mut faces = self.detect_faces(rgb, width, height)?;
        let mut pending: Vec<usize> = people
            .iter()
            .enumerate()
            .filter(|(_, person)| person.y2 - person.y1 >= 36.0 && !faces.iter().any(|face| inside(person, *face)))
            .map(|(index, _)| index)
            .collect();
        pending.sort_by(|&a, &b| {
            let area = |index: usize| (people[index].x2 - people[index].x1) * (people[index].y2 - people[index].y1);
            area(b).total_cmp(&area(a))
        });
        pending.truncate(4);
        for index in pending {
            let person = &people[index];
            let (crop, origin_x, origin_y, crop_w, crop_h) = head_crop(rgb, width, height, person);
            if crop_w < 16 || crop_h < 16 {
                continue;
            }
            let mut local = self.detect_faces(&crop, crop_w, crop_h)?;
            for face in &mut local {
                face[0] += origin_x;
                face[1] += origin_y;
                for point in 0..5 {
                    face[4 + point * 2] += origin_x;
                    face[5 + point * 2] += origin_y;
                }
            }
            faces.extend(local);
        }
        let faces = nms_faces(faces, 0.3);
        for person in people.iter_mut() {
            let Some(face) = faces.iter().find(|face| inside(person, **face)) else {
                continue;
            };
            person.face = Some(*face);
            if face[14] < 0.52 || face[2].min(face[3]) < 16.0 {
                continue;
            }
            if let Some(embedding) = self.embed_face(rgb, width, height, face)? {
                person.embedding = Some(embedding);
            }
        }
        Ok(())
    }

    fn detect_faces(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<[f32; 15]>> {
        const NET: u32 = 640;
        let scale = NET as f32 / width.max(height) as f32;
        let resized_w = ((width as f32 * scale).round() as u32).clamp(1, NET);
        let resized_h = ((height as f32 * scale).round() as u32).clamp(1, NET);
        let pad_x = (NET - resized_w) / 2;
        let pad_y = (NET - resized_h) / 2;
        let mut bgr = vec![0.0f32; 3 * (NET as usize) * (NET as usize)];
        for y in 0..resized_h {
            let sy = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, height as f32 - 1.0) as u32;
            for x in 0..resized_w {
                let sx = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, width as f32 - 1.0) as u32;
                let pixel = ((sy * width + sx) * 3) as usize;
                for channel in 0..3usize {
                    let value = rgb[pixel + (2 - channel)] as f32;
                    let dst = (channel * NET as usize + (y + pad_y) as usize) * NET as usize + (x + pad_x) as usize;
                    bgr[dst] = value;
                }
            }
        }
        let array = Array4::from_shape_vec((1, 3, NET as usize, NET as usize), bgr)?;
        let outputs = self.detect.run(ort::inputs![TensorRef::from_array_view(&array)?])?;
        let mut faces = Vec::new();
        for stride in [8i32, 16, 32] {
            let cols = (NET as i32) / stride;
            let rows = cols;
            let cls = named(&outputs, &format!("cls_{stride}"))?;
            let obj = named(&outputs, &format!("obj_{stride}"))?;
            let bbox = named(&outputs, &format!("bbox_{stride}"))?;
            let kps = named(&outputs, &format!("kps_{stride}"))?;
            for row in 0..rows {
                for col in 0..cols {
                    let index = (row * cols + col) as usize;
                    let score = (cls[index].clamp(0.0, 1.0) * obj[index].clamp(0.0, 1.0)).sqrt();
                    if score < 0.5 {
                        continue;
                    }
                    let stride_f = stride as f32;
                    let cx = (col as f32 + bbox[index * 4]) * stride_f;
                    let cy = (row as f32 + bbox[index * 4 + 1]) * stride_f;
                    let bw = bbox[index * 4 + 2].exp() * stride_f;
                    let bh = bbox[index * 4 + 3].exp() * stride_f;
                    let mut face = [0.0; 15];
                    face[0] = (cx - bw * 0.5 - pad_x as f32) / scale;
                    face[1] = (cy - bh * 0.5 - pad_y as f32) / scale;
                    face[2] = bw / scale;
                    face[3] = bh / scale;
                    for point in 0..5 {
                        let kx = (kps[index * 10 + point * 2] + col as f32) * stride_f;
                        let ky = (kps[index * 10 + point * 2 + 1] + row as f32) * stride_f;
                        face[4 + point * 2] = (kx - pad_x as f32) / scale;
                        face[5 + point * 2] = (ky - pad_y as f32) / scale;
                    }
                    face[14] = score;
                    if face[2] < 12.0 || face[3] < 12.0 {
                        continue;
                    }
                    faces.push(face);
                }
            }
        }
        Ok(nms_faces(faces, 0.3))
    }

    fn embed_face(&mut self, rgb: &[u8], width: u32, height: u32, face: &[f32; 15]) -> Result<Option<Vec<f32>>> {
        let landmarks = [
            [face[4], face[5]],
            [face[6], face[7]],
            [face[8], face[9]],
            [face[10], face[11]],
            [face[12], face[13]],
        ];
        let matrix = similarity(&REF, &landmarks);
        let mut tensor = vec![0.0f32; 3 * 112 * 112];
        for y in 0..112 {
            for x in 0..112 {
                let sx = matrix[0] * x as f32 + matrix[1] * y as f32 + matrix[2];
                let sy = matrix[3] * x as f32 + matrix[4] * y as f32 + matrix[5];
                let pixel = sample(rgb, width, height, sx, sy);
                for channel in 0..3 {
                    tensor[(channel * 112 + y) * 112 + x] = pixel[channel];
                }
            }
        }
        let array = Array4::from_shape_vec((1, 3, 112, 112), tensor)?;
        let outputs = self.embed.run(ort::inputs![TensorRef::from_array_view(&array)?])?;
        let mut values: Vec<f32> = outputs[0].try_extract_array::<f32>()?.iter().copied().collect();
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for value in &mut values {
                *value /= norm;
            }
        }
        Ok(Some(values))
    }
}

fn named(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<Vec<f32>> {
    let value = outputs
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("YuNet 缺少输出 {name}"))?;
    Ok(value.try_extract_array::<f32>()?.iter().copied().collect())
}

fn head_crop(rgb: &[u8], width: u32, height: u32, person: &Person) -> (Vec<u8>, f32, f32, u32, u32) {
    let bw = (person.x2 - person.x1).max(1.0);
    let bh = (person.y2 - person.y1).max(1.0);
    let x1 = (person.x1 - bw * 0.12).clamp(0.0, width as f32 - 1.0) as u32;
    let y1 = (person.y1 - bh * 0.08).clamp(0.0, height as f32 - 1.0) as u32;
    let x2 = (person.x2 + bw * 0.12).clamp(0.0, width as f32) as u32;
    let y2 = (person.y1 + bh * 0.72).clamp(0.0, height as f32) as u32;
    let crop_w = x2.saturating_sub(x1).max(1);
    let crop_h = y2.saturating_sub(y1).max(1);
    let mut crop = vec![0u8; (crop_w * crop_h * 3) as usize];
    for row in 0..crop_h {
        let src = (((y1 + row) * width + x1) * 3) as usize;
        let dst = (row * crop_w * 3) as usize;
        let len = (crop_w * 3) as usize;
        if src + len <= rgb.len() {
            crop[dst..dst + len].copy_from_slice(&rgb[src..src + len]);
        }
    }
    (crop, x1 as f32, y1 as f32, crop_w, crop_h)
}

fn inside(person: &Person, face: [f32; 15]) -> bool {
    let cx = face[0] + face[2] * 0.5;
    let cy = face[1] + face[3] * 0.5;
    person.x1 - 8.0 <= cx && cx <= person.x2 + 8.0 && person.y1 - 8.0 <= cy && cy <= person.y2 + 8.0
}

fn nms_faces(mut faces: Vec<[f32; 15]>, thresh: f32) -> Vec<[f32; 15]> {
    faces.sort_by(|a, b| b[14].total_cmp(&a[14]));
    let mut kept = Vec::new();
    for face in faces {
        if kept.iter().all(|other: &[f32; 15]| face_iou(&face, other) < thresh) {
            kept.push(face);
        }
    }
    kept
}

fn face_iou(a: &[f32; 15], b: &[f32; 15]) -> f32 {
    let ax2 = a[0] + a[2];
    let ay2 = a[1] + a[3];
    let bx2 = b[0] + b[2];
    let by2 = b[1] + b[3];
    let inter = (ax2.min(bx2) - a[0].max(b[0])).max(0.0) * (ay2.min(by2) - a[1].max(b[1])).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn similarity(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [f32; 6] {
    let mut src_mean = [0.0, 0.0];
    let mut dst_mean = [0.0, 0.0];
    for index in 0..5 {
        src_mean[0] += src[index][0];
        src_mean[1] += src[index][1];
        dst_mean[0] += dst[index][0];
        dst_mean[1] += dst[index][1];
    }
    src_mean[0] /= 5.0;
    src_mean[1] /= 5.0;
    dst_mean[0] /= 5.0;
    dst_mean[1] /= 5.0;
    let mut variance = 0.0;
    let mut parallel = 0.0;
    let mut cross = 0.0;
    for index in 0..5 {
        let sx = src[index][0] - src_mean[0];
        let sy = src[index][1] - src_mean[1];
        let dx = dst[index][0] - dst_mean[0];
        let dy = dst[index][1] - dst_mean[1];
        variance += sx * sx + sy * sy;
        parallel += sx * dx + sy * dy;
        cross += sx * dy - sy * dx;
    }
    let scale = if variance < 1e-6 {
        1.0
    } else {
        (parallel * parallel + cross * cross).sqrt() / variance
    };
    let norm = (parallel * parallel + cross * cross).sqrt().max(1e-6);
    let cos = parallel / norm;
    let sin = cross / norm;
    let r00 = scale * cos;
    let r01 = scale * -sin;
    let r10 = scale * sin;
    let r11 = scale * cos;
    let tx = dst_mean[0] - (r00 * src_mean[0] + r01 * src_mean[1]);
    let ty = dst_mean[1] - (r10 * src_mean[0] + r11 * src_mean[1]);
    [r00, r01, tx, r10, r11, ty]
}

fn sample(rgb: &[u8], width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
    if width == 0 || height == 0 {
        return [0.0; 3];
    }
    let x = x.clamp(0.0, width as f32 - 1.0);
    let y = y.clamp(0.0, height as f32 - 1.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut pixel = [0.0; 3];
    for channel in 0..3usize {
        let at = |sx: u32, sy: u32| rgb[((sy * width + sx) * 3 + channel as u32) as usize] as f32;
        pixel[channel] = at(x0, y0) * (1.0 - fx) * (1.0 - fy)
            + at(x1, y0) * fx * (1.0 - fy)
            + at(x0, y1) * (1.0 - fx) * fy
            + at(x1, y1) * fx * fy;
    }
    pixel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_maps_reference_center() {
        let shifted = REF.map(|point| [point[0] + 10.0, point[1] + 4.0]);
        let matrix = similarity(&REF, &shifted);
        let x = matrix[0] * 56.0 + matrix[1] * 71.0 + matrix[2];
        let y = matrix[3] * 56.0 + matrix[4] * 71.0 + matrix[5];
        assert!((x - 66.0).abs() < 0.5);
        assert!((y - 75.0).abs() < 0.5);
    }
}
