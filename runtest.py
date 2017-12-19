import os
import shutil
import src.utils as utils
import src.config as config
import src.networks as networks

def test_url_downloader():
    if os.path.exists(config.tmp_dir):
        shutil.rmtree(config.tmp_dir)
    os.makedirs(config.tmp_dir)
    dummyURL = 'https://raw.githubusercontent.com/tianhaoz95/pics/master/Profile_avatar_placeholder_large.png'
    utils.download_img(dummyURL, config.tmp_dir + '/test')

def test_dataset_updater():
    if config.clean_up_dataset and os.path.exists(config.dataset_dir):
        shutil.rmtree(config.dataset_dir)
    utils.update_dataset()
    utils.update_dataset()

def test_update_metadata():
    utils.update_metadata()

def test_img_loader():
    img_list = os.listdir(config.dataset_dir)
    img = utils.load_img(img_list[0], config.resolution)

def test_load_dataset():
    train_x, train_y, test_x, test_y = utils.load_dataset()

def clean_up():
    if config.clean_up_dataset:
        shutil.rmtree(config.dataset_dir)
    shutil.rmtree(config.tmp_dir)

def main():
    print('running tests ...')
    test_update_metadata()
    test_url_downloader()
    test_dataset_updater()
    test_img_loader()
    test_load_dataset()
    clean_up()

if __name__ == '__main__':
    try:
        main()
        print('all test cases passed')
    except:
        print('some test cases did not pass')
