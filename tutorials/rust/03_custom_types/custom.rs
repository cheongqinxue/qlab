#![allow(dead_code)] // Hides warnings for unused code

#[derive(Debug)]
struct Person {
    name: String,
    age: u8,
}

#[derive(Debug)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Debug)]
struct Rect {
    tl: Point,
    br: Point,
}

fn main() {
    let name = String::from("Adam");
    let age: u8 = 90;
    let teacher = Person{name, age};
    println!("Person: {:?}", teacher);

    let p1 = Point { x: 1.2, y: 3.4 };
    let p2 = Point { x: 3.2, y: 5.4 };

    println!("Coords: ({:?}), ({:?})", p1, p2);

    let p3 = Point { x:3.2, ..p2 };

    // Deconstruct a struct!
    let Point { x: x_, y: y_ } = p3;

    println!("P3 coords: x={}, y={}", x_, y_);

    
}