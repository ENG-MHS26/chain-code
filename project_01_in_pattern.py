import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve

def load_mnist():
    file = Path('mnist.npz')
    if not file.exists():
        print("Downloading MNIST dataset...")
        urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', file)

    with np.load(file) as data:
        images = np.concatenate([data['x_train'], data['x_test']])
        labels = np.concatenate([data['y_train'], data['y_test']])
    return images, labels

def split_balanced(labels):
    rng = np.random.default_rng(7)
    count = int(np.bincount(labels).min())
    count -= count % 5  # Allows exactly 80% training, 20% testing
    train, test = [], []

    for digit in range(10):
        ids = np.where(labels == digit)[0]
        rng.shuffle(ids)
        ids = ids[:count]
        cut = count * 4 // 5
        train.extend(ids[:cut])
        test.extend(ids[cut:])

    train, test = np.array(train), np.array(test)
    print('Training images:', len(train), 'Testing images:', len(test))
    print('Train per digit:', np.bincount(labels[train], minlength=10))
    print('Test per digit: ', np.bincount(labels[test], minlength=10))
    return train, test

def get_contour(image):
    binary = (image > 100).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return binary, None
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return binary, points

def make_chain_code(points):
    directions = [[3, 2, 1],
                  [4, -1, 0],
                  [5, 6, 7]]
    code = []

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        dx, dy = int(x2 - x1), int(y2 - y1)
        code.append(directions[dy + 1][dx + 1])
    return code
def normalize_code(code):
    first = [(code[i] - code[i - 1]) % 8 for i in range(len(code))]
    normal = first.copy()
    for i in range(1, len(first)):
        shifted = first[i:] + first[:i]
        if shifted < normal:
            normal = shifted
    return first, normal

def extract_features(images, indices):
    features = []
    for index in indices:
        _, points = get_contour(images[index])
        if points is None or len(points) < 2:
            features.append([])
            continue
        code = make_chain_code(points)
        _, normalized = normalize_code(code)
        features.append(normalized)
    return features

def show_samples(images, labels, train):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for digit, ax in enumerate(axes.flat):
        for index in train:
            if labels[index] == digit:
                break
        binary, points = get_contour(images[index])
        if points is None:
            continue
            
        code = make_chain_code(points)
        first, normal = normalize_code(code)

        ax.imshow(binary, cmap='gray')
        closed = np.vstack([points, points[0]])
        ax.plot(closed[:, 0], closed[:, 1], color='lime', linewidth=1.5)

        # Red arrows show Chain Code direction (مع تباعد بسيط لترتيب الشكل)
        step = max(1, len(points) // 15)  
        for i in range(0, len(points), step):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            ax.arrow(int(x1), int(y1), int(x2 - x1), int(y2 - y1),
                     color='red', alpha=0.8, width=0.012, head_width=0.3,
                     length_includes_head=True)

        # Yellow circle for starting point
        ax.scatter(points[0, 0], points[0, 1], color='yellow', s=40, zorder=5)
        ax.set_title(f'Digit {digit}')
        ax.axis('off')

        if digit == 0:
            print('\n--- Digit 0 Chain Code Sample ---')
            print('Chain Code:', code[:20], '...')
            print('First Difference:', first[:20], '...')
            print('Normalized Code:', normal[:20], '...')

    plt.tight_layout()
    plt.savefig('chaincode_result.png', dpi=150)
    print("\nVisual result saved as 'chaincode_result.png'")
    plt.show()


def main():
    images, labels = load_mnist()
    train, test = split_balanced(labels)
    train_features = extract_features(images, train)
    show_samples(images, labels, train) 

if __name__ == '__main__':
    main()