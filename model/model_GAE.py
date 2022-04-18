import torch
import numpy as np
from model import utils_GAE
#from metrics import cal_clustering_metric
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import scipy.io as scio
import pdb

class GAE(torch.nn.Module):
    def __init__(self, X, labels, layers=None, num_neighbors=5, learning_rate=10**-3,
                 max_iter=500, device=None):
        super(GAE, self).__init__()
        self.layers = layers
        if self.layers is None:
            self.layers = [X.shape[1], 256, 64]
        self.device = device
        if self.device is None:
            self.torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.labels = labels
        self.num_neighbors = num_neighbors
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._build_up()

    def _build_up(self):
        self.W1 = get_weight_initial([self.layers[0], self.layers[1]])
        self.W2 = get_weight_initial([self.layers[1], self.layers[2]])

    def forward(self, Laplacian):
        # sparse
        embedding = Laplacian.mm(self.X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        # sparse
        self.embedding = Laplacian.mm(embedding.matmul(self.W2))
        softmax = torch.nn.Softmax(dim=1)
        recons_w = self.embedding.matmul(self.embedding.t())
        recons_w = softmax(recons_w)
        return recons_w + 10**-10

    def build_loss(self, recons, weights):
        size = self.X.shape[0]
        loss = torch.norm(recons - weights, p='fro')**2 / size
        return loss

    def run(self):
        weights, _ = utils_GAE.cal_weights_via_CAN(self.X.t(), self.num_neighbors)
        _ = None
        Laplacian = utils_GAE.get_Laplacian_from_weights(weights)
        print('Raw-CAN:', end=' ')
        #self.clustering(weights, method=2, raw=True)
        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            recons = self(Laplacian)
            loss = self.build_loss(recons, weights)
            loss.backward()
            optimizer.step()
            # recons = None
            torch.cuda.empty_cache()
            if (i+1) % 50 == 0 or i == 0:
                print('Iteration-{}, loss={}, '.format(i+1, round(loss.item(), 5)), end=' ')
                #self.clustering((recons.abs() + recons.t().abs()).detach()/2, method=2)
        # self.clustering((recons.abs() + recons.t().abs()).detach()/2, method=2)

    def clustering(self, weights, method=2, raw=False):
        n_clusters = np.unique(self.labels).shape[0]
        if method == 0 or method == 2:
            embedding = self.X if raw else self.embedding
            embedding = embedding.cpu().detach().numpy()
            km = KMeans(n_clusters=n_clusters).fit(embedding)
            prediction = km.predict(embedding)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='     ')
        if method == 1 or method == 2:
            degree = torch.sum(weights, dim=1).pow(-0.5)
            L = (weights * degree).t() * degree
            L = L.cpu()
            _, vectors = L.symeig(True)
            indicator = vectors[:, -n_clusters:]
            indicator = indicator / (indicator.norm(dim=1) + 10**-10).repeat(n_clusters, 1).t()
            indicator = indicator.cpu().numpy()
            km = KMeans(n_clusters=n_clusters).fit(indicator)
            prediction = km.predict(indicator)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
        print('')


class AdaGAE(torch.nn.Module):
    def __init__(self, X, labels, layers=None, lam=0.1, num_neighbors=3, learning_rate=10**-3,
                 max_iter=50, max_epoch=10, update=True, inc_neighbors=2, links=0, device=None):
        super(AdaGAE, self).__init__()
        if layers is None:
            layers = [1024, 256, 64]
        if device is None:
            device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.X = torch.Tensor(X).to(device)
        #self.labels = labels
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.num_neighbors = num_neighbors + 1
        self.embedding_dim = layers[-1]
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.update = update
        self.inc_neighbors = inc_neighbors
        self.max_neighbors = self.cal_max_neighbors()
        self.links = links
        self.device = device

        self.embedding = None
        self._build_up()

    def _build_up(self):
        self.W1 = get_weight_initial([self.input_dim, self.mid_dim])
        self.W2 = get_weight_initial([self.mid_dim, self.embedding_dim])

    def cal_max_neighbors(self):
        if not self.update:
            return 0
        size = self.X.shape[0]
        num_clusters = 20 #np.unique(self.labels).shape[0]
        return 1.0 * size / num_clusters

    def forward(self, Laplacian):
        # sparse
        embedding = Laplacian.mm(self.X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        # sparse
        self.embedding = Laplacian.mm(embedding.matmul(self.W2))
        distances = utils_GAE.distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        # sparseProb = SparseProb(sparsity=self.num_neighbors)
        # recons_w = sparseProb(distances)
        return recons_w + 10**-10
        # return 1 / (distances + 1)
        # return torch.sigmoid(self.embedding.matmul(torch.t(self.embedding)))

    def update_graph(self):
        weights, raw_weights = utils_GAE.cal_weights_via_CAN(self.embedding.t(), self.num_neighbors, self.links)  # first
        weights = weights.detach()
        raw_weights = raw_weights.detach()
        # threshold = 0.5
        # connections = (recons > threshold).type(torch.IntTensor).cuda()
        # weights = weights * connections
        Laplacian = utils_GAE.get_Laplacian_from_weights(weights)
        # Laplacian = utils_GAE.get_Laplacian_from_weights(utils_GAE.noise(weights))
        return weights, Laplacian, raw_weights

    def update_graph_entropy(self, recons):
        size = self.embedding.shape[0]
        distances = utils_GAE.distance(self.embedding.t(), self.embedding.t())
        distances = self.lam * distances - recons.log()
        distances = distances.detach()
        distances = torch.exp(-distances)
        sorted_distances, _ = distances.sort(dim=1, descending=True)
        flags = sorted_distances[:, self.num_neighbors].reshape([size, 1])
        distances[distances <= flags] = 0
        sums = distances.sum(dim=1).reshape([size, 1])
        raw_weights = distances / sums
        weights = (raw_weights + raw_weights.t()) / 2
        Laplacian = utils_GAE.get_Laplacian_from_weights(weights)
        return weights, Laplacian, raw_weights

    def build_loss(self, recons, weights, raw_weights):
        size = self.X.shape[0]
        loss = 0
        loss += raw_weights * torch.log(raw_weights / recons + 10**-10)
        loss = loss.sum(dim=1)
        loss = loss.mean()
        # loss += 10**-3 * (torch.mean(self.embedding.pow(2)))
        # loss += 10**-3 * (torch.mean(self.W1.pow(2)) + torch.mean(self.W2.pow(2)))
        # loss += 10**-3 * (torch.mean(self.W1.abs()) + torch.mean(self.W2.abs()))
        degree = weights.sum(dim=1)
        L = torch.diag(degree) - weights
        loss += self.lam * torch.trace(self.embedding.t().matmul(L).matmul(self.embedding)) / size
        return loss

    def run(self):
        weights, raw_weights = utils_GAE.cal_weights_via_CAN(self.X.t(), self.num_neighbors, self.links)  
        # weights = torch.where(
        #         torch.isnan(weights), 
        #         torch.full_like(weights, 0), 
        #         weights)
        # raw_weights = torch.where(
        #         torch.isnan(raw_weights), 
        #         torch.full_like(raw_weights, 0), 
        #         raw_weights)
        #pdb.set_trace()
        Laplacian = utils_GAE.get_Laplacian_from_weights(weights)
        Laplacian = Laplacian.to_sparse()
        torch.cuda.empty_cache()
        # Laplacian = utils_GAE.get_Laplacian_from_weights(utils_GAE.noise(weights))
        print('Raw-CAN:', end=' ')
        #self.clustering(weights, k_means=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()
                recons = self(Laplacian)
                #pdb.set_trace()
                loss = self.build_loss(recons, weights, raw_weights)
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                torch.cuda.empty_cache()
                loss.backward()
                optimizer.step()
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
                #print('epoch-%3d-i:%3d,' % (epoch, i), 'loss: %6.5f' % loss.item())
            # scio.savemat('results/embedding_{}.mat'.format(epoch), {'Embedding': self.embedding.cpu().detach().numpy()})
            if self.num_neighbors < self.max_neighbors:
                weights, Laplacian, raw_weights = self.update_graph()
                # weights, Laplacian, raw_weights = self.update_graph_entropy(recons)
                # self.clustering(weights, k_means=False)
                #acc, nmi = self.clustering(weights, k_means=True, SC=True)
                self.num_neighbors += self.inc_neighbors
            else:
                if self.update:
                    self.num_neighbors = int(self.max_neighbors)
                    break
                recons = None
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                torch.cuda.empty_cache()
                w, _, __ = self.update_graph()
                _, __ = (None, None)
                torch.cuda.empty_cache()
                # w, _, __ = self.update_graph_entropy(recons)
                #acc, nmi = self.clustering(w, k_means=False)
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
                if self.update:
                    break
            print('epoch:%3d,' % epoch, 'loss: %6.5f' % loss.item())
        #return acc, nmi
        return weights

    def clustering(self, weights, k_means=True, SC=True):
        n_clusters = np.unique(self.labels).shape[0]
        if k_means:
            embedding = self.embedding.cpu().detach().numpy()
            km = KMeans(n_clusters=n_clusters).fit(embedding)
            prediction = km.predict(embedding)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end=' ')
        if SC:
            degree = torch.sum(weights, dim=1).pow(-0.5)
            L = (weights * degree).t() * degree
            L = L.cpu()
            _, vectors = L.symeig(True)
            indicator = vectors[:, -n_clusters:]
            indicator = indicator / (indicator.norm(dim=1)+10**-10).repeat(n_clusters, 1).t()
            indicator = indicator.cpu().numpy()
            km = KMeans(n_clusters=n_clusters).fit(indicator)
            prediction = km.predict(indicator)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
        print('')
        return acc, nmi


class SparseProb(torch.nn.Module):
    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, distances):
        """
        :param distances: n * n
        :return:
        """
        size = distances.shape[0]
        sorted_distances, _ = distances.sort(dim=1)
        top_k = sorted_distances[:, self.sparsity]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

        sum_top_k = torch.sum(sorted_distances[:, 0:self.sparsity], dim=1)
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))
        prob = torch.div(top_k - distances, self.sparsity * top_k - sum_top_k)
        return prob.relu()


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    import data_loader as loader
    dataset = loader.MNIST_TEST
    data, labels = loader.load_data(dataset)
    mDevice = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    input_dim = data.shape[1]
    X = torch.Tensor(data).to(mDevice)
    if dataset is loader.USPS:
        layers = [input_dim, 128, 64]
    elif dataset is loader.TOY_THREE_RINGS:
        layers = [input_dim, 3, 2]
    elif dataset is loader.SEGMENT:
        layers = [input_dim, 10, 7]
    elif dataset is loader.GLASS:
        layers = [input_dim, 7, 5]
    elif dataset is loader.CORA:
        layers = [input_dim, 1600, 400]
    else:
        layers = [input_dim, 256, 64]
    for neighbor in [5, 10, 20]:
        gae = GAE(X, labels, layers=layers, num_neighbors=neighbor, learning_rate=10**-3, max_iter=200, device=mDevice)
        gae.run()

