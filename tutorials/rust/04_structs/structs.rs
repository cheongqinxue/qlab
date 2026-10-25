#[derive(Debug)]
struct User {
    name: String,
    active: bool,
    signin_count: i32,
    email: String,
}

impl User {
    fn toggle_active_state(&mut self) {
        self.active = !self.active;
    }
}

fn main() {
    // Plain initialization
    let mut user = User {
        name: String::from("John"),
        active: true,
        signin_count: 0i32,
        email: String::from("john@hotmail.com"),
    };

    println!("01. user={:?}", user);

    // Changing a value
    user.email = String::from("john0002@hotmail.com");
    println!("02. Changing a value:\n\tuser={:?}", user);

    // Creating instance with struct update syntax
    let mut user2 = User {
        email: String::from("somebody@yahoo.com"),
        ..user
    };
    println!(
        "03. Creating instance with struct update syntax:\n\tuser2={:?}",
        user2
    );
    // println!(
    //     "03. Creating instance with struct update syntax:\n\tuser={:?}\n\tuser2={:?}",
    //     user, user2
    // );
    // NOTE: The above creates the following compile error:
    // s$ rustc structs.rs
    // error[E0382]: borrow of partially moved value: `user`
    // --> structs.rs:31:9
    // |
    // 25 |       let user2 = User {
    // |  _________________-
    // 26 | |         email: String::from("somebody@yahoo.com"),
    // 27 | |         ..user
    // 28 | |     };
    // | |_____- value partially moved here
    // ...
    // 31 |           user, user2
    // |           ^^^^ value borrowed here after partial move
    // |
    // = note: partial move occurs because `user.name` has type `String`, which does not implement
    // the `Copy` trait
    // error: aborting due to 1 previous error
    // Whats happening here is that the struct update syntaxed moved data from user to user2, so
    // ownership of some data has changed hands. That partial ownership transfer has rendered
    // user no longer usable. What you could have done is:
    // ```rust
    // let user2 = User {
    //     email: String::from("somebody@yahoo.com"),
    //     --user.clone(),
    // }
    // ```

    // Using a method
    user2.toggle_active_state();
    println!("04. Using a method:\n\tuser2={:?}", user2);
}
