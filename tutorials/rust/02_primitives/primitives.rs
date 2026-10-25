#![allow(dead_code)]

use std::mem;

fn reverse(pair: (i32, bool)) -> (bool, i32) {
    let (a, b) = pair;
    (b, a)
}

fn analyze_slice(slice: &[i32]) {
    println!("first number: {:?}", slice[0]);
    println!("number of numbers: {}", slice.len());
}

#[derive(Debug)]
struct Matrix(f32, f32, f32);

fn main() {
    // Type annotated
    let logical: bool = true;

    let a_float: f64 = 1.0;
    let an_integer = 5i32; //suffix annotation

    // using a default
    let default_float = 3.0; // f64
    let default_integer = 1; // i32

    // inferred from context
    let mut inferred = 12;
    inferred = 123123124124i64; // i64 inferred from this line

    // mutables
    let mut mutable = 12;
    mutable = 11;

    // Type of mutable cannot be changed
    // mutable = true;
    // $ rustc primitives.rs
    // error[E0308]: mismatched types
    // --> primitives.rs:21:15
    // |
    // 17 |     let mut mutable = 12;
    // |                       -- expected due to this value
    // ...
    // 21 |     mutable = true;
    // |               ^^^^ expected integer, found `bool`

    // error: aborting due to 1 previous error

    // For more information about this error, try `rustc --explain E0308`.

    // Compound types
    let array: [i32; 5] = [1, 2, 3, 4, 5];

    // Tuple is a collection values of different types
    let pair = (3i32, true);

    println!("pair = {:?}", pair);

    let rev_pair = reverse(pair);

    println!("reversed pair = {:?}", rev_pair);

    let mat = Matrix(1.1, 1.2, 3.3);

    println!("matrix = {:?}", mat);

    let arr: [i32; 3] = [0, 1, 3];

    println!("Size of array: {} bytes", mem::size_of_val(&arr));

    analyze_slice(&arr);

    let empty: [i32; 0] = [];
    assert_eq!(&empty, &[]);

    for i in 0..arr.len() + 1 {
        match arr.get(i) {
            Some(xval) => println!("{}: {}", i, xval),
            None => println!("{} is too far!", i),
        }
    }
}
