from jigsaw.dataset.dataset import build_test_dataloader
import hydra
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../config", config_name="global_config")
def main(cfg):
    # cfg.data.batch_size = 1
    test_loader = build_test_dataloader(cfg)
    scale_values = []

    for data_dict in tqdm(test_loader):
        scale = data_dict["part_scale"]
        part_valids = data_dict['part_valids'].bool()
        scale = scale[part_valids]

        for j in range(scale.shape[0]):
            c_scale = scale[j].item()
            if c_scale < 0.1:
                scale_values.append(c_scale)
        

    # Plotting the distribution
    plt.hist(scale_values, bins=30)  # Adjust the number of bins as needed
    plt.title('Distribution of Scale Values')
    plt.xlabel('Scale')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    main()