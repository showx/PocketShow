use super::track::Person;

pub struct TargetLock {
    lost_timeout_s: f64,
    locked_id: Option<i64>,
    locked_person: Option<String>,
    manual: bool,
    missing_s: f64,
}

impl TargetLock {
    pub fn new(lost_timeout_s: f64) -> Self {
        Self {
            lost_timeout_s,
            locked_id: None,
            locked_person: None,
            manual: false,
            missing_s: 0.0,
        }
    }

    pub fn locked_id(&self) -> Option<i64> {
        self.locked_id
    }

    pub fn clear(&mut self) {
        *self = Self::new(self.lost_timeout_s);
    }

    pub fn lock(&mut self, person: &Person) {
        self.locked_id = Some(person.id);
        self.locked_person = person.person_id.clone();
        self.manual = true;
        self.missing_s = 0.0;
    }

    pub fn lock_at(&mut self, x: f32, y: f32, people: &[Person]) -> Option<i64> {
        let mut hits: Vec<&Person> = people
            .iter()
            .filter(|person| person.x1 <= x && x <= person.x2 && person.y1 <= y && y <= person.y2)
            .collect();
        hits.sort_by(|a, b| area(b).total_cmp(&area(a)));
        let person = hits.first()?;
        self.lock(person);
        Some(person.id)
    }

    pub fn cycle(&mut self, people: &[Person], step: i32) -> Option<i64> {
        if people.is_empty() {
            return self.locked_id;
        }
        let mut ordered = people.to_vec();
        ordered.sort_by_key(|person| person.id);
        let index = self
            .locked_id
            .and_then(|id| ordered.iter().position(|person| person.id == id))
            .map(|index| (index as i32 + step).rem_euclid(ordered.len() as i32) as usize)
            .unwrap_or(0);
        self.lock(&ordered[index]);
        Some(ordered[index].id)
    }

    pub fn update<'a>(&mut self, people: &'a [Person], dt: f64) -> Option<&'a Person> {
        if let Some(id) = self.locked_id {
            if let Some(current) = people.iter().find(|person| person.id == id) {
                if current.person_id.is_some() {
                    self.locked_person = current.person_id.clone();
                }
                self.missing_s = 0.0;
                return Some(current);
            }
        }
        if let Some(person_id) = self.locked_person.clone() {
            if let Some(found) = people
                .iter()
                .filter(|person| person.person_id.as_deref() == Some(person_id.as_str()))
                .max_by(|a, b| (area(a) * a.conf).total_cmp(&(area(b) * b.conf)))
            {
                self.locked_id = Some(found.id);
                self.missing_s = 0.0;
                return Some(found);
            }
        }
        if self.locked_id.is_some() {
            self.missing_s += dt;
            if self.missing_s < self.lost_timeout_s || self.manual {
                return None;
            }
            self.locked_id = None;
            self.locked_person = None;
            self.missing_s = 0.0;
            self.manual = false;
        }
        let chosen = people.iter().max_by(|a, b| (area(a) * a.conf).total_cmp(&(area(b) * b.conf)));
        if let Some(person) = chosen {
            self.locked_id = Some(person.id);
            self.locked_person = person.person_id.clone();
            self.missing_s = 0.0;
        }
        chosen
    }
}

fn area(person: &Person) -> f32 {
    (person.x2 - person.x1).max(0.0) * (person.y2 - person.y1).max(0.0)
}
