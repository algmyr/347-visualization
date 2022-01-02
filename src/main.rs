use std::f64::consts;

use anyhow::{Context, Result};
use cairo::{Format, ImageSurface};
use clap::Parser;
use itertools::Itertools;
use num::Integer;

type Point = num::Complex<f64>;

#[derive(Clone, Copy)]
struct Rect {
  x_min: f64,
  x_max: f64,
  y_min: f64,
  y_max: f64,
}

#[allow(dead_code)]
impl Rect {
  fn x_center(&self) -> f64 { (self.x_max + self.x_min) / 2.0 }
  fn y_center(&self) -> f64 { (self.y_max + self.y_min) / 2.0 }
  fn width(&self) -> f64 { self.x_max - self.x_min }
  fn height(&self) -> f64 { self.y_max - self.y_min }
}

#[derive(Clone, Copy)]
struct Color {
  r: f64,
  g: f64,
  b: f64,
  a: f64,
}

#[allow(dead_code)]
impl Color {
  fn rgba(r: f64, g: f64, b: f64, a: f64) -> Self { Self { r, g, b, a } }
  fn rgb(r: f64, g: f64, b: f64) -> Self { Self::rgba(r, g, b, 1.0) }
  fn gray(x: f64) -> Self { Self::rgba(x, x, x, 1.0) }
}

trait CairoExtensions {
  fn set_color(&self, color: Color);
  fn circle(&self, xc: f64, yc: f64, radius: f64);
  fn move_to_c(&self, z: Point);
  fn line_to_c(&self, z: Point);
  fn polygon_c(&self, zs: &Vec<Point>);
}

impl CairoExtensions for cairo::Context {
  fn set_color(&self, color: Color) { self.set_source_rgba(color.r, color.g, color.b, color.a); }
  fn circle(&self, xc: f64, yc: f64, radius: f64) { self.arc(xc, yc, radius, 0.0, consts::TAU) }
  fn polygon_c(&self, zs: &Vec<Point>) {
    self.move_to_c(zs[zs.len() - 1]);
    for &z in zs {
      self.line_to_c(z);
    }
  }
  fn move_to_c(&self, z: Point) { self.move_to(z.re, z.im) }
  fn line_to_c(&self, z: Point) { self.line_to(z.re, z.im) }
}

fn normalize_rotation(mut g: Vec<Point>) -> usize {
  let snap = |x: f64| (x * 1024.0).round() / 1024.0;
  let snap_c = |z: &Point| (snap(z.re), snap(z.im));
  let mn = *g
    .iter()
    .min_by(|x, y| snap_c(x).partial_cmp(&snap_c(*y)).unwrap())
    .unwrap();
  let mut i = 0;
  while g[0] != mn {
    g.rotate_left(1);
    i += 1;
  }
  i
}

fn setup_cairo(scale: f64) -> Result<(ImageSurface, cairo::Context)> {
  let margin = 0.2;
  let rect = Rect {
    x_min: -1.0 - margin,
    x_max: 1.0 + margin,
    y_min: -1.0 - margin,
    y_max: 1.0 + margin,
  };
  let width = (scale * rect.width()) as i32;
  let height = (scale * rect.height()) as i32;
  let surface =
    ImageSurface::create(Format::ARgb32, width, height).context("Surface creation failed")?;
  let context = cairo::Context::new(&surface).context("Context creation failed")?;
  context.scale(scale, scale);
  context.translate(-rect.x_min, -rect.y_min);
  context.scale(1.0, -1.0);
  Ok((surface, context))
}

fn compute_shapes(points: &Vec<Point>, n_poly: usize, n_skip: usize) -> Vec<Vec<Point>> {
  let n_small_polys = n_poly - n_skip;
  let mut shapes = vec![];
  for poly_i in 0..n_small_polys {
    let g = (0..n_skip)
      .map(|i| points[(poly_i + n_small_polys * i) % points.len()])
      .collect_vec();

    fn circumference(g: &Vec<num::Complex<f64>>) -> f64 {
      let mut dist = 0.0;
      let mut last = g[g.len() - 1];
      for p in g {
        dist += (last - *p).norm_sqr();
        last = *p;
      }
      dist
    }

    // Apparently extracting points like this can give stars.
    // I haven't figured out the conditions for this.
    // Just pick the ordering with minimum circumference.
    let g = (1..((n_skip + 1) / 2).max(2))
      .map(|d| (0..n_skip).map(|i| g[d * i % g.len()]).collect_vec())
      .min_by(|a, b| circumference(a).partial_cmp(&circumference(b)).unwrap())
      .unwrap();

    shapes.push(g);
  }

  shapes
}

fn uniform(start: f64, end: f64, steps: usize) -> impl Iterator<Item = f64> {
  let delta = (end - start) / (steps as f64);
  (0..steps).map(move |i| start + (i as f64) * delta)
}

fn expi(x: f64) -> Point { (x * Point::i()).exp() }

fn inner_radius(n_skip: usize, n_poly: usize) -> f64 {
  // inner circle spins at rate t/r
  // n_poly small rotations should complete in time n_skip
  // n_skip/r = n_poly
  // r = n_skip/n_poly
  (n_skip as f64) / (n_poly as f64)
}
fn correction(r: f64) -> f64 {
  // Correction to make the segment straightish
  // at the cost of not being on the unit circle anymore.
  // This puts the center on the segment on the polygon line.
  let segment_angle = 2.0 * consts::PI * r;
  let circle_segment_height = 1.0 - (segment_angle / 2.0).cos();
  2.0 * r - circle_segment_height
}

fn draw(
  scale: f64,
  star_path: &Vec<num::Complex<f64>>,
  small_polys: Vec<Vec<num::Complex<f64>>>,
  big_poly: Vec<num::Complex<f64>>,
  points: Vec<num::Complex<f64>>,
) -> Result<ImageSurface, anyhow::Error> {
  let white = Color::gray(1.0);
  let black = Color::gray(0.0);
  let poly1_color = Color::rgba(0.0, 0.8, 0.0, 0.6);
  let poly2_color = Color::rgba(0.0, 0.0, 1.0, 0.6);
  let poly_line_width = 3.0;
  let circle_line_width = 4.0;
  let star_line_width = 1.0;
  let dot_size = 3.0;
  let (surface, context) = setup_cairo(scale)?;
  let pixel = 1.0 / scale;
  context.save()?;
  context.set_color(white);
  context.paint()?;
  context.set_color(black);
  context.set_line_width(circle_line_width * pixel);
  context.circle(0.0, 0.0, 1.0);
  context.stroke()?;
  context.set_color(black);
  context.set_line_width(star_line_width * pixel);
  context.polygon_c(star_path);
  context.stroke()?;
  context.restore()?;
  for poly in &small_polys {
    context.save()?;
    context.set_color(poly1_color);
    context.set_line_width(poly_line_width * pixel);
    context.polygon_c(poly);
    context.stroke()?;
    context.restore()?;
  }
  context.save()?;
  context.set_color(poly2_color);
  context.set_line_width(poly_line_width * pixel);
  context.polygon_c(&big_poly);
  context.stroke()?;
  context.restore()?;
  for p in &points {
    context.save()?;
    context.set_color(black);
    context.circle(p.re, p.im, dot_size * pixel);
    context.fill()?;
    context.restore()?;
  }
  Ok(surface)
}

fn produce(
  n_skip: usize,
  n_poly: usize,
  n_frames: usize,
  duration: f64,
  scale: f64,
  mut collector: gifski::Collector,
) -> Result<()> {
  let star_f = |t: f64| {
    // n_poly sided polygon
    // n_skip full circle revolutions
    let r = inner_radius(n_skip, n_poly);
    let corr = correction(r);
    let rotation_rad = 1.0 - r;
    expi(t) * (rotation_rad + expi(-t / r) * (r - corr))
  };

  let star_length = (n_skip as f64) * consts::TAU;
  let star_path = uniform(0.0, star_length, n_poly * 30)
    .map(star_f)
    .collect_vec();

  let n_points = (n_poly - n_skip) * n_skip;
  let shape_normalization_offsets = {
    let points = uniform(0.0, 1.0, n_points)
      .map(|norm_t| star_f(norm_t * star_length))
      .collect_vec();
    compute_shapes(&points, n_poly, n_skip)
      .into_iter()
      .map(normalize_rotation)
      .collect_vec()
  };

  for (i, (dt, frame_time)) in Iterator::zip(
    uniform(0.0, 1.0 / (n_points / n_skip) as f64, n_frames),
    uniform(0.0, duration, n_frames),
  )
  .enumerate()
  {
    // Points of interest are equally spaced around the path.
    let points = uniform(0.0, 1.0, n_points)
      .map(|norm_t| star_f((norm_t + dt) * star_length))
      .collect_vec();

    let small_polys = Iterator::zip(
      compute_shapes(&points, n_poly, n_skip).into_iter(),
      &shape_normalization_offsets,
    )
    .map(|(mut shape, &offset)| {
      shape.rotate_left(offset);
      shape
    })
    .collect_vec();

    let big_poly = small_polys.iter().map(|g| g[0]).collect_vec();

    let surface = draw(scale, &star_path, small_polys, big_poly, points)?;

    let height = surface.height() as usize;
    let width = surface.width() as usize;
    let img_data = surface
      .take_data()?
      .chunks(4)
      .map(|rgba| match rgba {
        &[b, g, r, a] => rgb::RGBA8 { r, g, b, a },
        _ => panic!("Badness"),
      })
      .collect_vec();
    collector.add_frame_rgba(i, imgref::Img::new(img_data, height, width), frame_time)?;
  }
  Ok(())
}

fn do_stuff(args: Args) -> Result<()> {
  if args.n_skip.gcd(&args.n_poly) != 1 {
    return Err(anyhow::Error::msg("Arguments not relatively prime"));
  }

  let n_frames =
    ((args.fps as f64) * args.duration) as usize;

  let settings = gifski::Settings {
    width: None,
    height: None,
    quality: 100,
    fast: true,
    repeat: gifski::Repeat::Infinite,
  };
  let (collector, writer) = gifski::new(settings)?;

  let writer_thread = std::thread::spawn(|| -> Result<()> {
    let f = std::fs::File::create(args.outfile).context("Opening output image failed")?;
    writer
      .write(f, &mut gifski::progress::NoProgress {})
      .context("Gifski write failed")?;
    Ok(())
  });

  let producer_thread = std::thread::spawn(move || -> Result<()> {
    produce(
      args.n_skip,
      args.n_poly,
      n_frames,
      args.duration,
      args.scale,
      collector,
    )
  });

  producer_thread.join().unwrap()?;
  writer_thread.join().unwrap()?;

  Ok(())
}

//

#[derive(Parser)]
#[clap(about, version, author)]
struct Args {
  #[clap(long, default_value_t = 30)]
  fps: i32,
  #[clap(long, default_value_t = 3.0)]
  duration: f64,
  #[clap(long, default_value_t = 200.0)]
  scale: f64,
  #[clap(short, long, default_value = "out.gif")]
  outfile: String,
  #[clap(required = true)]
  n_poly: usize,
  #[clap(required = true)]
  n_skip: usize,
}

fn main() -> Result<()> { do_stuff(Args::parse()) }
