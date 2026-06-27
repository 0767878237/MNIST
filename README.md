# MNIST Classification with PyTorch

Repo nay duoc dinh huong de hoc bai toan phan loai chu so viet tay MNIST theo huong hoc thuat toan, va framework duoc chon la `PyTorch`.

Thay vi chi tap trung vao viec "chay duoc code", README nay uu tien tra loi 3 cau hoi co ban:

1. Du lieu MNIST thuc chat trong ra sao?
2. Mo hinh hoc tu du lieu do bang cach nao?
3. Tai sao pipeline PyTorch lai duoc thiet ke nhu vay?

## 1. Bai toan dang can giai quyet

Muc tieu la xay dung mot mo hinh nhan vao mot anh xam kich thuoc `28 x 28` pixel va du doan anh do la mot chu so trong `10` lop:

`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

Day la bai toan:

- `supervised learning`: vi du lieu da co nhan
- `multi-class classification`: vi co 10 lop va moi anh chi thuoc 1 lop
- `image classification`: vi dau vao la anh

## 2. Du lieu hien co trong repo

Thu muc [data](C:/Users/Dell/Desktop/MNIST/data) dang chua 4 file goc cua MNIST:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Y nghia cua 4 file nay:

- `train-images`: tap anh de huan luyen
- `train-labels`: nhan cua tap train
- `t10k-images`: tap anh de kiem tra
- `t10k-labels`: nhan cua tap test

Trong MNIST:

- Tap train co `60,000` anh
- Tap test co `10,000` anh
- Moi anh la ma tran `28 x 28`
- Moi pixel co gia tri trong do xam tu `0` den `255`

## 3. Truc giac cua bai toan

Hay tuong tuong moi anh `28 x 28` la mot bang so gom `784` gia tri pixel.

Mo hinh se co gang hoc quy luat:

- cum pixel nao thuong xuat hien trong chu so `0`
- net cong nao thuong xuat hien trong chu so `3`
- duong thang doc nao thuong lien quan den chu so `1`

Noi cach khac, mo hinh khong "hieu" chu so nhu con nguoi, ma no hoc mot ham so:

`image -> class score -> predicted digit`

## 4. Flow hoc thuat toan cua bai toan

Toan bo bai toan co the hieu qua 6 buoc.

### Buoc 1. Nap du lieu tu file `.idx`

MNIST trong repo dang o dang raw binary `.idx`, nen buoc dau tien la doc file va bien no thanh tensor.

Sau buoc nay, ta mong muon thu duoc:

- `X_train`: tensor chua anh train
- `y_train`: tensor chua nhan train
- `X_test`: tensor chua anh test
- `y_test`: tensor chua nhan test

Trong PyTorch, du lieu cuoi cung thuong duoc dua ve dang:

- anh: `torch.float32`
- nhan: `torch.long`

Ly do:

- `float32` phu hop cho tinh toan gradient
- `long` la kieu PyTorch yeu cau cho `CrossEntropyLoss`

### Buoc 2. Kiem tra du lieu de tranh hoc sai ngay tu dau

Truoc khi train, can kiem tra:

- shape cua `X_train`, `X_test`
- so luong nhan cua tung lop
- mot vai anh mau va nhan di kem

Day la buoc rat quan trong ve mat hoc thuat toan. Neu doc sai du lieu, moi ket qua train sau do deu vo nghia.

Vi du, can xac nhan:

- train images co shape gan dung `60000 x 28 x 28`
- test images co shape gan dung `10000 x 28 x 28`
- nhan nam trong khoang `0-9`

### Buoc 3. Tien xu ly de mo hinh hoc on dinh

Du lieu goc co pixel trong khoang `0-255`, nhung mo hinh hoc tot hon khi du lieu da duoc scale.

Buoc tien xu ly thuong gom:

- Chuan hoa pixel: `x = x / 255.0`
- Them channel: tu `(N, 28, 28)` thanh `(N, 1, 28, 28)` cho CNN
- Tach validation set tu training set

Tai sao phai chuan hoa?

- Gia tri du lieu nho va dong deu hon
- Gradient on dinh hon
- Toc do hoc thuong tot hon

Tai sao phai tach validation?

- De theo doi kha nang tong quat hoa
- De phat hien overfitting som

### Buoc 4. Bieu dien bai toan duoi dang mo hinh hoc may

Sau khi co tensor dau vao, bai toan duoc viet lai thanh:

- Dau vao: anh `x`
- Dau ra: vector score gom `10` gia tri
- Nhan dung: `y` trong khoang `0-9`

Voi PyTorch, mo hinh se xuat ra `logits`, vi du:

```text
[2.1, -0.4, 0.8, 5.7, 1.2, ...]
```

Moi gia tri la do "tu tin chua chuan hoa" cua mo hinh voi tung lop.

Lop co score cao nhat se la du doan cuoi cung.

### Buoc 5. Ham mat mat va qua trinh hoc

Day la phan trung tam cua thuat toan.

#### 5.1. Ham mat mat

Voi bai toan MNIST da lop, lua chon chuan la:

- `nn.CrossEntropyLoss()`

Ly do:

- Phu hop cho phan loai nhieu lop
- Nhan truc tiep `logits`
- Nhan dung dang so nguyen `0-9`

Ham mat mat se phat cao khi:

- mo hinh du doan sai
- hoac du doan dung nhung do tu tin thap

Muc tieu cua train la lam loss giam dan.

#### 5.2. Toi uu hoa

Qua trinh hoc dien ra theo vong lap:

1. Dua mot batch anh vao model
2. Tinh `logits`
3. Tinh `loss`
4. `backward()` de tinh gradient
5. `optimizer.step()` de cap nhat trong so

Neu viet ngan gon theo PyTorch:

```python
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

Y nghia hoc thuat toan:

- `forward`: mo hinh dua ra du doan
- `loss`: do muc do sai
- `backward`: tinh xem nen sua trong so theo huong nao
- `step`: cap nhat trong so de lan sau du doan tot hon

### Buoc 6. Danh gia mo hinh

Sau khi hoc xong, khong danh gia tren tap train ma danh gia tren tap test.

Nhung chi so quan trong:

- `accuracy`
- `confusion matrix`
- cac anh du doan sai

#### Accuracy

Accuracy cho biet ti le du doan dung:

`accuracy = so mau dung / tong so mau`

Day la chi so de hieu nhat voi MNIST.

#### Confusion matrix

Confusion matrix tra loi cau hoi:

- lop nao de bi nham nhat
- mo hinh nham `5` thanh `3` nhieu hay nham `5` thanh `8`

Day la buoc bien con so accuracy thanh hieu biet thuc su ve hanh vi cua model.

## 5. Tai sao nen bat dau bang baseline truoc?

Ve mat hoc thuat toan, baseline la moc so sanh dau tien.

Nen di theo thu tu:

1. `Softmax / Logistic Regression`
2. `MLP`
3. `CNN`

Ly do:

- Neu baseline khong hoc duoc, co the pipeline dang loi
- Neu baseline da on, ta moi co co so de noi CNN cai thien den dau

## 6. Vi tri cua PyTorch trong bai toan

PyTorch giup ta trien khai toan bo pipeline hoc may:

- `Dataset`: dong goi du lieu
- `DataLoader`: chia batch va shuffle
- `nn.Module`: dinh nghia mo hinh
- `loss function`: do sai so
- `optimizer`: cap nhat tham so
- `train loop`: lap qua nhieu epoch
- `eval loop`: danh gia khong tinh gradient

Flow PyTorch co the tom tat nhu sau:

```text
IDX files
   ->
Custom Dataset / tensor conversion
   ->
DataLoader
   ->
PyTorch model
   ->
CrossEntropyLoss
   ->
Backward propagation
   ->
Optimizer step
   ->
Evaluation on test set
```

## 7. Cau truc code nen huong toi

Neu trien khai bang PyTorch, repo nen phat trien theo cau truc sau:

```text
MNIST/
|-- data/
|-- notebooks/
|-- src/
|   |-- data_loader.py
|   |-- dataset.py
|   |-- preprocess.py
|   |-- train.py
|   |-- evaluate.py
|   |-- models/
|   |   |-- mlp.py
|   |   |-- cnn.py
|-- outputs/
|   |-- figures/
|   |-- checkpoints/
|-- README.md
```

## 8. Scaffold da duoc trien khai

Repo hien da co bo khung PyTorch toi thieu:

- [src/data_loader.py](C:/Users/Dell/Desktop/MNIST/src/data_loader.py:1): doc file `.idx`
- [src/preprocess.py](C:/Users/Dell/Desktop/MNIST/src/preprocess.py:1): chuan hoa va tach validation
- [src/dataset.py](C:/Users/Dell/Desktop/MNIST/src/dataset.py:1): `Dataset` va `DataLoader`
- [src/models/mlp.py](C:/Users/Dell/Desktop/MNIST/src/models/mlp.py:1): baseline `MLP`
- [src/models/cnn.py](C:/Users/Dell/Desktop/MNIST/src/models/cnn.py:1): model `CNN`
- [src/train.py](C:/Users/Dell/Desktop/MNIST/src/train.py:1): train loop chinh
- [src/evaluate.py](C:/Users/Dell/Desktop/MNIST/src/evaluate.py:1): evaluate va confusion matrix
- [src/visualize.py](C:/Users/Dell/Desktop/MNIST/src/visualize.py:1): luu history va ve bieu do loss/accuracy

## 9. Cach chay

### Cai dat thu vien

```bash
pip install -r requirements.txt
```

### Train baseline MLP

```bash
python -m src.train --model mlp --epochs 5
```

### Train CNN

```bash
python -m src.train --model cnn --epochs 5
```

### Mot vai tham so co the doi

```bash
python -m src.train --model cnn --optimizer adam --batch-size 128 --learning-rate 0.001 --epochs 10
```

Sau khi train:

- mo hinh tot nhat se duoc luu trong `outputs/checkpoints/`
- ket qua in ra gom `train loss`, `train accuracy`, `validation loss`, `validation accuracy`
- cuoi cung se co `test accuracy` va `confusion matrix`
- lich su train se duoc luu trong `outputs/figures/<model>_history.json`
- bieu do `loss/accuracy` se duoc luu trong `outputs/figures/<model>_training_curves.png`

### Doc bieu do nhu the nao

- Neu `train loss` giam va `validation loss` cung giam, model dang hoc on.
- Neu `train accuracy` tang rat cao nhung `validation accuracy` tang cham hoac giam, model co dau hieu overfit.
- Neu ca train va validation deu dung yen tu som, model co the dang qua don gian hoac learning rate chua phu hop.

## 10. Ke hoach trien khai bang PyTorch

Phan nay duoc viet lai de phu hop voi dinh huong PyTorch va uu tien hieu thuat toan.

# Plan

Xay dung mot pipeline MNIST bang PyTorch de di tu du lieu `.idx` goc den mot baseline co the huan luyen, danh gia, va giai thich duoc hanh vi cua mo hinh. Cach tiep can uu tien hieu ban chat tung buoc trong thuat toan truoc khi toi uu ket qua.

## Scope
- In: Doc du lieu MNIST tu file `.idx`, chuyen thanh tensor PyTorch, tien xu ly, huan luyen baseline, danh gia va giai thich cach hoc cua mo hinh.
- Out: Deployment, giao dien, distributed training, hyperparameter tuning quy mo lon.

## Action items
[ ] Tao bo doc file `data/*.idx` va chuyen du lieu thanh `torch.Tensor` dung kieu.
[ ] Kiem tra shape, phan bo nhan, va mot so anh mau de xac nhan du lieu hop le.
[ ] Chuan hoa pixel va reshape ve `(N, 1, 28, 28)` de san sang cho CNN trong PyTorch.
[ ] Tach tap validation tu training set va dong goi du lieu bang `Dataset` va `DataLoader`.
[ ] Xay dung baseline `MLP` don gian bang `torch.nn.Module` de xac minh pipeline hoc.
[ ] Huan luyen bang `CrossEntropyLoss` va mot optimizer nhu `SGD` hoac `Adam`.
[ ] Theo doi `train loss`, `train accuracy`, `validation loss`, `validation accuracy` theo tung epoch.
[ ] Danh gia tren tap `t10k` bang `test accuracy`, confusion matrix, va phan tich cac mau nham lan.
[ ] Trien khai them mot `CNN` co ban va so sanh voi `MLP` de thay loi ich cua dac trung khong gian.
[ ] Tong hop ket qua thanh nhan xet hoc thuat toan: model hoc duoc gi, sai o dau, va vi sao.

## Open questions
- Ban muon minh uu tien code baseline `MLP` truoc hay di thang sang `CNN`?
- Ban co muon them notebook de vua hoc vua chay thu khong?
- Ban co muon minh scaffold luon ma nguon PyTorch theo plan nay khong?

## 11. Dieu can nho nhat

Neu chi giu lai mot y, thi do la:

`MNIST khong chi la bai toan code model, ma la bai toan hieu cach du lieu di qua loss, gradient, va cap nhat trong so de bien anh thanh du doan.`

Flow cot loi la:

`doc du lieu -> tensor hoa -> chuan hoa -> model -> loss -> backward -> update -> evaluate`

Nam chac flow nay, ban se hieu duoc nen tang cua rat nhieu bai toan deep learning khac, khong chi rieng MNIST.
