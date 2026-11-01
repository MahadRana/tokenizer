## 11/01/2026
A1: `train` borrows the corpus as `&str` and takes `num_merges`, instead of reading a path and always doing 1000 merges. `Merges` and `Vocab` are type aliases for the two hash maps. The loop is unchanged: count pairs, take `top_pair`, merge, record, and break when every remaining pair occurs once. `main` still does the file read, asks for 1000 merges, and prints when training stops early. Tests call `train` directly, so they no longer need temp files. The missing-file panic stays on `read_text`. `book.txt` (Pride and Prejudice) is gitignored.

## 10/03/2026
Got the training loop written: count pairs, merge top pair, then save pair -> ID rule in Encoder list and update Decoder hashmap with new ID, (ID, \[first's bytes, seconds' bytes\])

Obstacles: 
- Can't return a borrow of something the func owns, it got freed
- Can't index strings cause UTF-8 can be more than 1 byte, use as_bytes insead
- .iter() hands me references, use * to get vals
- map[&key] crashes if the key's missing, .get() gives an Option instead

## 10/05/2026
Wrote encode. Until a pair is not found in whole string, we look through the code, find the minimum ID pair using the encoder HashMap, replace all the values with that, then reloop. return that Vec\<u32\>

Wrote decoder. For the tokens, just replace them with the values from the decoder, then convert it to a String using String::from_utf8() converted

Obstacles: 
- contains_key and get both take references not the values themselves
- when doing a for each loop, the id is already a reference
- Result Type
    - Ok(T) -> good
    - Err(E) -> error
    - Multiple ways to handle
        - unwrap() or expect("msg")to panic on err
            - Panic = crash program, print error, exit
            - Best for quick experiments
        - unwrap_or(default)
            - dont panic, just return default on err
        - ? (BEST)
            - if value is good, then give value, else return early with Err
            - requires Result\<CorrectType, ErrorType\> return type
                - ex from code Result\<String, std::string::FromUtf8Error\>
        
