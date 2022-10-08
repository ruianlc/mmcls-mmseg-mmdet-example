from mmcls.apis import inference_model, init_model, show_result_pyplot
import mmcv

from util.base_util import plt, os, get_project_rootpath


def main():
    # Specify the path to config file and checkpoint file
    config_file = r'configs/myconfig_vit.py'
    #checkpoint_file = r'checkpoints/epoch_10.pth'
    checkpoint_file = r'data/output/epoch_20.pth'
    
    # checkpoint_file = 'checkpoints/resnext50_32x4d_batch256_imagenet_20200708-c07adbb7.pth'
    # Specify the device. You may also use cpu by `device='cpu'`.
    device = 'cpu'
    # Build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device=device)

    # Test a single image
    img = os.path.join(get_project_rootpath(), 'data/input/cifar-10/test/airplane/0000.jpg')
    #img = r'data/input/cifar-10/test/airplane/0000.jpg'
    result = inference_model(model, img)

    # Show the results
    show_result_pyplot(model, img, result)
    plt.show()
    aa=0

if __name__ == '__main__':
    main()