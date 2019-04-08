import ann

def main():
	train_x, train_y, test_x, test_y = ann.load_data_set("dataSet.csv")
	model = ann.fit(train_x, train_y)
	print(ann.evaluate(test_x, test_y, model, "model"))
	
if __name__ == '__main__':
	main()