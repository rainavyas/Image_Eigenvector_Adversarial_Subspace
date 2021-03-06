import torch
import torch.nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_handler import get_datasets
from training_tools import *
from torch.nn import CosineSimilarity
from attack_tools import *

def get_fooling_rate(test_loader, attack_model, trained_model):
    fool = AverageMeter()

    # Switch to eval mode
    attack_model.eval()
    trained_model.eval()

    with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):

      input_var = input.to(device)

      # compute output without attack
      output_original = trained_model(input_var)
      class_original = torch.argmax(output_original, dim=1)

      # compute output with attack
      output_attacked = attack_model(input_var, trained_model)

      # Get fooling rate - the percentage of classes changed
      acc = accuracy_topk(output_attacked.data, class_original.data, k=1)
      curr_fool = 100 - acc
      fool.update(curr_fool.item(), input.size(0))
    return fool.avg

def get_covariance_matrix(dataset):
    train_loader_no_batch = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    # get all the training images into a tensor [num_images x chn*IMG_DIM*IMG_DIM]
    dataiter = iter(train_loader_no_batch)
    images, labels = dataiter.next()
    X = torch.reshape(images, (images.size(0), -1))

    # compute covariance matrix
    X_mean = torch.mean(X, dim=0)
    X_mean_matrix = torch.outer(X_mean, X_mean)
    X_corr_matrix = torch.matmul(torch.transpose(X, 0, 1), X)/X.size(0)

    Cov = X_corr_matrix - X_mean_matrix
    return Cov

def get_eigenvectors_and_eigenvalues(Cov):
    e, v = torch.symeig(Cov, eigenvectors=True)
    v = torch.transpose(v, 0, 1)
    e_abs = torch.abs(e)

    inds = torch.argsort(e_abs, descending=True)
    e = e[inds]
    v = v[inds]

    return e, v

def get_avg_kl(test_loader, attack_model, trained_model):
    """
    Calculate KL divergance between pmf before and after each sample being attacked
    Return average kl divergence across all samples
    """
    kl_div = AverageMeter()

    # switch to evaluate mode
    attack_model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute no attack output
            sf = torch.nn.Softmax(dim=1)
            pmf_no_attack = sf(trained_model(input_var))
            pmf_attacked = torch.log(sf(attack_model(input_var, trained_model))) # Log for kl div function

            # Compute kl divergence
            kl_mean = torch.nn.functional.kl_div(pmf_attacked, pmf_no_attack, reduction='batchmean')
            kl_div.update(kl_mean.item(), input.size(0))

    return kl_div.avg

def get_average_comps_squared(e, v, attack_models):
    '''
    For individual attacks
    '''

    cosine_totals = torch.zeros(e.size(0))
    counter = torch.zeros(e.size(0))

    ranks = []
    cosines_sq = []
    cosines_sq_cum = []

    cum=0

    for i in range(e.size(0)):
        curr_e = e[i]
        curr_v = v[i]
        cos = CosineSimilarity(dim=1)

        counter = 0
        cosine_total = 0

        for attack_model in attack_models:
          attack_vectors = torch.FloatTensor(attack_model.attack.cpu()).to(device)
          attack_vectors = torch.reshape(attack_vectors, (attack_vectors.size(0), -1))
          v_repeat = curr_v.repeat(attack_vectors.size(0), 1).to(device)
          abs_cosines_sq = torch.abs(cos(attack_vectors, v_repeat))**2
          summed = torch.sum(abs_cosines_sq, dim=0).item()
          cosine_total += summed
          counter += attack_vectors.size(0)

        cosine_avg = cosine_total/counter
        cum += cosine_avg
        ranks.append(i)
        cosines_sq.append(cosine_avg)
        cosines_sq_cum.append(cum)

    return ranks, cosines_sq, cosines_sq_cum

def get_r_f_e(trained_model, e, v, stepsize=40, epsilon=0.1, chn=3, IMG_DIM=32):
    # report fooling rate: percentage of points that change class with attack
    ranks = []
    fools = []
    eigenvalues = []

    for i in range(0, e.size(0), stepsize):
    ranks.append(i)
    eigenvalues.append(e[i])
    attack_direction = v[i]

    attack_signs = torch.sign(attack_direction)
    attack = (attack_signs * epsilon) # can multiply by -1 to reverse direction of attack
    attack = torch.reshape(attack, (chn, IMG_DIM, IMG_DIM))

    # Evaluate impact of attack
    attack_init = torch.zeros(chn, IMG_DIM, IMG_DIM)
    hvsm_attack = Attack(attack_init)
    hvsm_attack.to(device)

    old_params = {}
    for name, params in hvsm_attack.named_parameters():
      old_params[name] = params.clone()

    old_params['attack'] = attack
    for name, params in hvsm_attack.named_parameters():
      params.data.copy_(old_params[name])

    # Get fooling rate
    fool = get_fooling_rate(test_loader, hvsm_attack, trained_model)
    fools.append(fool)
    return ranks, fools, eigenvalues

def get_r_kl_e(trained_model, e, v, stepsize=40, epsilon=0.01, chn=3, IMG_DIM=32):
    ranks = []
    kls=[]
    eigenvalues = []

    for i in range(0, e.size(0), stepsize):
        curr_e = e[i]
        curr_v = v[i]
        attack_signs = torch.sign(curr_v)
        attack = (attack_signs * epsilon) # can multiply by -1 to reverse direction of attack
        attack_init = torch.reshape(attack, (chn, IMG_DIM, IMG_DIM))
        attack_model = Attack(attack_init)
        attack_model.to(device)
        kl = get_avg_kl(test_loader, attack_model, trained_model)
        ranks.append(i)
        eigenvalues.append(curr_e)
        kls.append(kl)
    return ranks, kls, eigenvalues

def get_area_non_overlap(cum1, cum2):
    '''
    Area non-overlap for cumulative plots
    '''
    total = 0
    for c1, c2 in zip(cum1, cum2):
        diff = abs(c2-c1)
        total += diff
    return total
