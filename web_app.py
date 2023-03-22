from import_framework import *
from load_model import *

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def predict_label():
    if request.method == "GET":
        return render_template("home.html", value="Image")
    if request.method == "POST":
        if "file" not in request.files:
            return "Hình ảnh không được tải lên!"
        
        file = request.files["file"].read()

        try:
            img = Image.open(io.BytesIO(file))
        except IOError:
            return jsonify(predictions = "Đây không phải là Hình ảnh, vui lòng tải lại tệp!")

    transform = BaseTransform(resize, mean, std)   # Tạo ra transform
    img_transformed = transform(img)   # Transform ảnh input

    # Chuyển đổi: (chanels, height, width) -> (height, width, channels)
    img_transformed = img_transformed.numpy().transpose(1,2,0)
    img_transformed = np.clip(img_transformed, 0, 1)

    class_index = json.load(open('./imagenet_class_index.json', 'r'))
    
    predictor = Predictor(class_index)

    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img)

    # Chuyển đổi tensor (3, 224, 224) sang tensor (1, 3, 224, 224)    
    img_transformed = img_transformed.unsqueeze_(0)

    # Dự đoán ảnh đầu vào bằng mạng neural
    out = net(img_transformed)
    result = predictor.predict_max(out)   # Tìm ra lớp có chỉ số chính xác cao nhất

    return jsonify(predictions = result)

if __name__ == '__main__':
    app.run(debug=True)