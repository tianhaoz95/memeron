import src.config as config
import src.utils as utils

def main():
    f = open(config.add_samples_dir, 'r')
    content = f.readlines()
    sample_type = content[0].strip()
    if sample_type != 'meme' and sample_type != 'nonmeme':
        print('wrong sample type')
        return
    url_list = content[1:]
    utils.add_metadata(url_list, sample_type)
    f.close()
    print('overwriting the file ...')
    overwrite = open(config.add_samples_dir, 'w')
    overwrite.write('fill in urls')
    overwrite.close()
    utils.update_dataset()

if __name__ == '__main__':
    main()
