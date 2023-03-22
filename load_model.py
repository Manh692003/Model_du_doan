from import_framework import *
import warnings

# Tắt cảnh báo liên quan đến pretrained và weights
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.", category=UserWarning)
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.", category=UserWarning)

resize = 224
mean = (0.485, 0.456, 0.406) 
std = (0.229, 0.224, 0.225)    

# Load các parameter(tham số) đã train vào net
net = models.vgg16(pretrained=True)
net.eval()   # Đặt mô hình vào chế độ đánh giá

# Định nghĩa một phương thức chuyển đổi (transform) dữ liệu ảnh
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    def __call__(self, img):
        return self.base_transform(img)

# Dự đoán nhãn của hình ảnh
class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    # Dự đoán nhãn của ảnh đầu vào
    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predict_label_name = self.class_index[str(maxid)]
        return predict_label_name