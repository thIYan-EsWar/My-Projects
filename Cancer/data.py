from random import shuffle, sample
with open('data.txt', 'r') as f:
	contents = f.readlines()
	contents = sample(contents, len(contents))
with open('train_data.txt', 'w') as f:
	[f.write(content) for content in contents[: 601]]
with open('test_data.txt', 'w') as f:
	[f.write(content) for content in contents[601:]]