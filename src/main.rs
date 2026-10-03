mod counter;

fn main() {
    let count = counter::create_counter("./book.txt");
    for pair in &count {
        let ((l1, l2), value) = pair;
        println!("({}, {}) {}", l1, l2, value);
    }
}
