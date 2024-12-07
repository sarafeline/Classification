import torch
import torchvision.transforms as T
import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageDraw
import os
import shutil
import random

# Local paths for images and annotations
image_dir = r"D:\Courses\IAV\CNN\COCO\train\train"
annotation_path = r"D:\Courses\IAV\CNN\COCO\filtered_annotations.json"
new_image_path = r"D:\Courses\IAV\CNN\cat14.png"

# Charger les annotations COCO
coco = COCO(annotation_path)

# Identifier les catégories 'cat'
cat_ids = coco.getCatIds(catNms=['cat'])
image_ids = coco.getImgIds(catIds=cat_ids)

# Créer un dataset personnalisé
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_path, transforms=None):
        self.root = root
        self.coco = COCO(annotation_path)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # Charger l'image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # Charger les boîtes
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']
            # Convertir [x, y, width, height] en [x_min, y_min, x_max, y_max]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # 1 correspond à "cat"

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Appliquer les transformations
transform = T.Compose([
    T.ToTensor()
])

# Fixer la graine pour la reproductibilité
random.seed(42)
torch.manual_seed(42)

# Créer le dataset et le diviser en training/validation
full_dataset = COCODataset(image_dir, annotation_path, transforms=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Charger le modèle Faster R-CNN pré-entraîné
model = models.fasterrcnn_resnet50_fpn(pretrained=True)

# Geler les couches pré-entraînées
for param in model.parameters():
    param.requires_grad = False

# Modifier le modèle pour détecter une seule classe (cat)
num_classes = 2  # 1 classe (cat) + 1 pour l'arrière-plan
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Déplacer le modèle sur le GPU si disponible
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Définir l'optimiseur et la fonction de perte
optimizer = torch.optim.SGD(model.roi_heads.box_predictor.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

# Fonction pour entraîner le modèle
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# Entraîner le modèle
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, optimizer, train_loader, device)

print("Entraînement terminé.")

# Fonction pour détecter les chats dans une image et les afficher
def detect_and_display(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Faire une prédiction
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Filtrer les prédictions pour garder uniquement les chats
    draw = ImageDraw.Draw(image)
    for i, label in enumerate(predictions['labels']):
        if label.item() == 1:  # La classe 1 correspond à "cat"
            score = predictions['scores'][i].item()
            if score > 0.5:  # Seulement si la confiance est > 50%
                box = predictions['boxes'][i].tolist()
                draw.rectangle(box, outline="red", width=3)

    # Afficher l'image avec les détections
    image.show()
# Save the image if a save path is provided
    if save_path:
        image.save(save_path)
        print(f"Image saved at {save_path}")

# Example usage:
save_path = r"D:\Courses\IAV\CNN\catso.png"
# Détecter et afficher les résultats sur la nouvelle image
detect_and_display(new_image_path, model, device)