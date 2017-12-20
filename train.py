import src.utils as utils
import src.networks as networks

def main():
	print("running main ...")
	utils.update_dataset()
	train_x, train_y, test_x, test_y = utils.load_dataset()
	# utils.check_data(train_x, train_y, 5)
	# utils.check_data(test_x, test_y, 5)
	networks.train_model(train_x, train_y, test_x, test_y)

if __name__ == "__main__":
	main()
