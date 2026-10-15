use anyhow::{anyhow, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

#[derive(Clone, Debug)]
pub struct Det {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
}

pub struct Yolo {
    session: Session,
    size: u32,
    conf: f32,
    iou: f32,
    input_name: String,
    tile: usize,
    seat: usize,
    held: Vec<(Det, u8)>,
}

impl Yolo {
    pub fn load(env: &ort::environment::Environment, path: &std::path::Path, size: u32, conf: f32, iou: f32) -> Result<Self> {
        let session = Session::builder(env)?.commit_from_file(path)?;
        let input_name = session.inputs()[0].name().to_string();
        let model_size = match session.inputs()[0].dtype() {
            ort::value::ValueType::Tensor { shape, .. } if shape.len() >= 4 && shape[2] > 0 && shape[2] == shape[3] => {
                shape[2] as u32
            }
            _ => size.max(32) / 32 * 32,
        };
        if model_size != size {
            eprintln!("检测模型输入是 {model_size}，按这个尺寸跑");
        }
        Ok(Self {
            session,
            size: model_size,
            conf,
            iou,
            input_name,
            tile: 0,
            seat: 0,
            held: Vec::new(),
        })
    }

    pub fn reset(&mut self) {
        self.tile = 0;
        self.seat = 0;
        self.held.clear();
    }

    pub fn detect(&mut self, rgb: &[u8], width: u32, height: u32, seats: &[(f32, f32, f32, f32)]) -> Result<Vec<Det>> {
        let mut found = self.infer(rgb, width, height, 0, 0, width, height, self.conf)?;
        let windows = tile_windows(width, height);
        if !windows.is_empty() {
            let (x1, y1, x2, y2) = windows[self.tile % windows.len()];
            self.tile += 1;
            found.extend(self.infer(rgb, width, height, x1, y1, x2, y2, self.conf)?);
        }
        if !seats.is_empty() {
            let (nx1, ny1, nx2, ny2) = seats[self.seat % seats.len()];
            self.seat += 1;
            let x1 = (nx1 * width as f32) as u32;
            let y1 = (ny1 * height as f32) as u32;
            let x2 = ((nx2 * width as f32) as u32).clamp(x1 + 1, width);
            let y2 = ((ny2 * height as f32) as u32).clamp(y1 + 1, height);
            let mut extra = self.infer(rgb, width, height, x1, y1, x2, y2, self.conf.min(0.08))?;
            extra.retain(|det| {
                let bw = det.x2 - det.x1;
                let bh = det.y2 - det.y1;
                bh >= 16.0 && bw >= 12.0 && bw * bh >= 280.0
            });
            found.extend(extra);
        }
        for (_, age) in &mut self.held {
            *age = age.saturating_add(1);
        }
        self.held.retain(|(_, age)| *age <= 12);
        found.extend(self.held.iter().map(|(det, _)| det.clone()));
        let found = nms(found, self.iou);
        for det in &found {
            if !self.held.iter().any(|(held, _)| iou(held, det) > 0.5) {
                self.held.push((det.clone(), 0));
            }
        }
        if self.held.len() > 80 {
            self.held.sort_by(|a, b| b.0.conf.total_cmp(&a.0.conf));
            self.held.truncate(80);
        }
        Ok(found)
    }

    fn infer(&mut self, rgb: &[u8], frame_w: u32, frame_h: u32, x1: u32, y1: u32, x2: u32, y2: u32, conf: f32) -> Result<Vec<Det>> {
        let crop_w = x2.saturating_sub(x1).max(1);
        let crop_h = y2.saturating_sub(y1).max(1);
        let mut crop = vec![0u8; (crop_w * crop_h * 3) as usize];
        for row in 0..crop_h {
            let src = (((y1 + row) * frame_w + x1) * 3) as usize;
            let dst = (row * crop_w * 3) as usize;
            let len = (crop_w * 3) as usize;
            if src + len <= rgb.len() {
                crop[dst..dst + len].copy_from_slice(&rgb[src..src + len]);
            }
        }
        let (tensor, map) = letterbox(&crop, crop_w, crop_h, self.size);
        let array = Array4::from_shape_vec((1, 3, self.size as usize, self.size as usize), tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => TensorRef::from_array_view(&array)?])?;
        let preds = outputs[0].try_extract_array::<f32>()?;
        let shape = preds.shape().to_vec();
        let flat: Vec<f32> = preds.iter().copied().collect();
        let mut dets = decode_people(&flat, &shape, conf, self.iou, &map, crop_w, crop_h);
        for det in &mut dets {
            det.x1 = (det.x1 + x1 as f32).clamp(0.0, frame_w as f32);
            det.y1 = (det.y1 + y1 as f32).clamp(0.0, frame_h as f32);
            det.x2 = (det.x2 + x1 as f32).clamp(0.0, frame_w as f32);
            det.y2 = (det.y2 + y1 as f32).clamp(0.0, frame_h as f32);
        }
        Ok(dets)
    }
}

fn tile_windows(width: u32, height: u32) -> Vec<(u32, u32, u32, u32)> {
    let cols = 2u32;
    let rows = 2u32;
    let overlap = 0.2f32;
    let tile_w = width as f32 / cols as f32;
    let tile_h = height as f32 / rows as f32;
    let mut windows = Vec::new();
    for row in 0..rows {
        let y1 = if row == 0 { 0.0 } else { row as f32 * tile_h - tile_h * overlap };
        let y2 = if row + 1 == rows { height as f32 } else { (row + 1) as f32 * tile_h + tile_h * overlap };
        for col in 0..cols {
            let x1 = if col == 0 { 0.0 } else { col as f32 * tile_w - tile_w * overlap };
            let x2 = if col + 1 == cols { width as f32 } else { (col + 1) as f32 * tile_w + tile_w * overlap };
            windows.push((
                x1.max(0.0) as u32,
                y1.max(0.0) as u32,
                (x2 as u32).min(width),
                (y2 as u32).min(height),
            ));
        }
    }
    windows
}

#[derive(Clone, Copy)]
pub(crate) struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

fn letterbox(rgb: &[u8], width: u32, height: u32, size: u32) -> (Vec<f32>, Letterbox) {
    let size_f = size as f32;
    let scale = size_f / (width.max(height) as f32);
    let resized_w = (width as f32 * scale).round().max(1.0) as u32;
    let resized_h = (height as f32 * scale).round().max(1.0) as u32;
    let pad_x = (size - resized_w) / 2;
    let pad_y = (size - resized_h) / 2;
    let mut canvas = vec![114.0 / 255.0; (size * size * 3) as usize];
    for y in 0..resized_h {
        let src_y = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, height as f32 - 1.0);
        let y0 = src_y.floor() as u32;
        let y1 = (y0 + 1).min(height - 1);
        let fy = src_y - y0 as f32;
        for x in 0..resized_w {
            let src_x = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, width as f32 - 1.0);
            let x0 = src_x.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let fx = src_x - x0 as f32;
            for channel in 0..3 {
                let sample = |sx: u32, sy: u32| rgb[((sy * width + sx) * 3 + channel) as usize] as f32 / 255.0;
                let value = sample(x0, y0) * (1.0 - fx) * (1.0 - fy)
                    + sample(x1, y0) * fx * (1.0 - fy)
                    + sample(x0, y1) * (1.0 - fx) * fy
                    + sample(x1, y1) * fx * fy;
                let dst = ((channel as u32 * size + (y + pad_y)) * size + (x + pad_x)) as usize;
                canvas[dst] = value;
            }
        }
    }
    (
        canvas,
        Letterbox {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        },
    )
}

pub fn decode_people(
    data: &[f32],
    shape: &[usize],
    conf: f32,
    iou: f32,
    map: &Letterbox,
    frame_w: u32,
    frame_h: u32,
) -> Vec<Det> {
    let (rows, cols, row_major) = match shape {
        [_, channels, count] if *channels <= 128 && *count > *channels => (*channels, *count, false),
        [_, count, channels] if *channels <= 128 && *count > *channels => (*channels, *count, true),
        _ => return Vec::new(),
    };
    if rows < 5 {
        return Vec::new();
    }
    let mut found = Vec::new();
    for index in 0..cols {
        let at = |row: usize| -> f32 {
            if row_major {
                data[index * rows + row]
            } else {
                data[row * cols + index]
            }
        };
        let score = at(4);
        if score < conf {
            continue;
        }
        let cx = (at(0) - map.pad_x) / map.scale;
        let cy = (at(1) - map.pad_y) / map.scale;
        let w = at(2) / map.scale;
        let h = at(3) / map.scale;
        let x1 = (cx - w * 0.5).clamp(0.0, frame_w as f32);
        let y1 = (cy - h * 0.5).clamp(0.0, frame_h as f32);
        let x2 = (cx + w * 0.5).clamp(0.0, frame_w as f32);
        let y2 = (cy + h * 0.5).clamp(0.0, frame_h as f32);
        if x2 - x1 < 4.0 || y2 - y1 < 4.0 {
            continue;
        }
        found.push(Det { x1, y1, x2, y2, conf: score });
    }
    nms(found, iou)
}

fn nms(mut dets: Vec<Det>, iou_thresh: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    let mut kept = Vec::new();
    for det in dets {
        if kept.iter().all(|other: &Det| iou(&det, other) < iou_thresh) {
            kept.push(det);
        }
    }
    kept
}

fn iou(a: &Det, b: &Det) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

pub fn ensure_model(path: &std::path::Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(anyhow!(
            "没有找到 {}。在仓库里执行：.venv/bin/python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, simplify=False)\"，再把 yolo11n.onnx 拷到这个路径。",
            path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn person_channel_survives_nms() {
        let channels = 5;
        let count = 8;
        let mut data = vec![0.0; channels * count];
        let put = |data: &mut [f32], index: usize, cx: f32, cy: f32, w: f32, h: f32, score: f32| {
            data[0 * count + index] = cx;
            data[1 * count + index] = cy;
            data[2 * count + index] = w;
            data[3 * count + index] = h;
            data[4 * count + index] = score;
        };
        put(&mut data, 0, 100.0, 100.0, 40.0, 80.0, 0.9);
        put(&mut data, 1, 102.0, 100.0, 40.0, 80.0, 0.4);
        put(&mut data, 2, 20.0, 20.0, 10.0, 10.0, 0.2);
        let map = Letterbox { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        let dets = decode_people(&data, &[1, channels, count], 0.5, 0.5, &map, 200, 200);
        assert_eq!(dets.len(), 1);
        assert!(dets[0].conf > 0.8);
    }
}
