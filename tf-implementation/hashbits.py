hash_me = "hi, my name is rohan!"

def do_hash(hash_me):
	hash = ""
	for char in hash_me:
		char = not(~(chr(char) & chr(4)) >> (char - 3)) << chr(7)
		hash+=char
	return hash

print(do_hash(hash_me))