import argparse

def parser():
    parser = argparse.ArgumentParser(description='Video Summarization')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
    parser.add_argument('--data_root', default='/home/yilin/Data', 
        help='the directory to save the dataset')
    parser.add_argument('--IN_data', default='/qnap/data_archive/imagenet-1k',
        help='the directory to save the imagenet dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--model-path', type=str, default='./checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
    parser.add_argument('--affix', default='default', help='the affix for the save folder')

    # parameters for generating adversarial examples
    parser.add_argument('--epsilon', '-e', type=float, default=0.0157, 
        help='maximum perturbation of adversaries (4/255=0.0157)')
    parser.add_argument('--beta_unlabel', type=float, default=1.0,
                    help='coefficiant for unlabeled loss')
    parser.add_argument('--alpha', '-a', type=float, default=0.00784, 
        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', '-k', type=int, default=10, 
        help='maximum iteration when generating adversarial examples')

    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--num_classes', type=int, default=257, help='number of classes')
    parser.add_argument('--IN_batch_size', type=int, default=128, help='batch size of imagenet')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=130, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--logit_grad_norm_decay', type=float, default=100.0,
        help='the parameter of l2 restriction for logit_grad_norm')
    parser.add_argument('--LGNR_temp', type=float, default=10.0,
        help='the parameter of temperature in exp function')
    parser.add_argument('--LGNR_decay', type=float, default=0.1,
        help='decay LGNR')
    parser.add_argument('--LGNR_softmax_temp', type=float, default=0.1,
        help='temperature in LGNR')
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, 
        help='the parameter of l2 restriction for weights')
    parser.add_argument('--beta_trades', type=float, default=6.0, 
        help='parameter of trades')
    parser.add_argument('--lambda_IN', type=float, default=0.3,
        help='parameter of lambda IN')
    parser.add_argument('--lambda_twins', type=float, default=0.3,
        help='parameter of lambda twins')
    parser.add_argument('--lambda_KD_robust', type=float, default=0.3,
        help='parameter of lambda KD loss')
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--n_eval_step', type=int, default=100, 
        help='number of iteration per one evaluation')
    parser.add_argument('--n_checkpoint_step', type=int, default=4000, 
        help='number of iteration to save a checkpoint')
    parser.add_argument('--n_store_image_step', type=int, default=4000, 
        help='number of iteration to save adversaries')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
    
    parser.add_argument('--adv_train', action='store_true')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
