import torch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X=torch.from_numpy(X).float().cuda()
        self.Y=torch.from_numpy(Y).long().cuda()
    def __len__(self):
        return len(self.X)
    def __getitem__(self,ix):
        return self.X[ix],self.Y[ix]

mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"], mnist["target"]
X.shape, Y.shape

X_train, X_test, y_train, y_test = X[:60000] / 255., X[60000:] / 255., Y[:60000].astype(int), Y[60000:].astype(int)
dataset=Dataset(X_train,y_train)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=100,shuffle=True)

D_in, H, D_out = 784, 100, 10
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
model.to("cuda")

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

def evaluate(x):
    model.eval()
    y_pred = model(x)
    y_probas = softmax(y_pred)
    return torch.argmax(y_probas, axis=1)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

epochs = 10
log_each = 1
l = []
model.train()
for e in range(1, epochs+1): 
    _l=[]
    for x_b,y_b in dataloader:
        y_pred = model(x_b)
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")
        
y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
print(accuracy_score(y_test, y_pred.cpu().numpy()))


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img=X_test[8]*255
label=y_test[8]
print(label)
print(y_pred[8].cpu().numpy())

img = img.reshape(28, 28)
 
img_show(img)
