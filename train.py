import src.utils as utils
import src.networks as networks

def main():
	print("running main ...")
	utils.update_dataset()
	train_x, train_y, test_x, test_y = utils.load_dataset()
	networks.train_model(train_x, train_y, test_x, test_y)

if __name__ == "__main__":
	main()
