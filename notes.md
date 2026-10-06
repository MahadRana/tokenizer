## 10/03/2026
Got the training loop written: count pairs, merge top pair, then save pair -> ID rule in Encoder list and update Decoder hashmap with new ID, (ID, \[first's bytes, seconds' bytes\])

Obstacles: 
- Can't return a borrow of something the func owns, it got freed
- Can't index strings cause UTF-8 can be more than 1 byte, use as_bytes insead
- .iter() hands me references, use * to get vals
- map[&key] crashes if the key's missing, .get() gives an Option instead

## 10/05/2026

Obstacles: 
Wrote encode. Until a pair is not found, we look through the code, find the minimum ID pair using the encoder HashMap, replace all the values with that, then reloop. return that Vec\<u32\>
- contains_key and get both take references not the values themselves