import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:google_ml_kit/google_ml_kit.dart';

class FaceRecognitionService {
  static const double THRESHOLD = 0.7;
  static const int INPUT_SIZE = 112;
  static const int OUTPUT_SIZE = 128;

  Interpreter? _interpreter;
  late FaceDetector _faceDetector;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      print('Initializing FaceRecognitionService...');

      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset('assets/MobileFaceNet.tflite',
          options: options);

      _faceDetector = GoogleMlKit.vision.faceDetector(FaceDetectorOptions(
        enableLandmarks: true,
        enableClassification: true,
        minFaceSize: 0.15,
        performanceMode: FaceDetectorMode.accurate,
      ));

      if (_interpreter != null) {
        print('FaceRecognitionService initialized successfully');
        _isInitialized = true;
      } else {
        throw Exception('Failed to load TensorFlow Lite model');
      }
    } catch (e) {
      print('Error initializing FaceRecognitionService: $e');
      _isInitialized = false;
      rethrow;
    }
  }

  Future<bool> detectFace(File imageFile) async {
    if (!_isInitialized) {
      throw Exception('FaceRecognitionService is not initialized');
    }
    try {
      final inputImage = InputImage.fromFile(imageFile);
      final faces = await _faceDetector.processImage(inputImage);
      return faces.isNotEmpty;
    } catch (e) {
      print('Error during face detection: $e');
      return false;
    }
  }

  Future<List<double>> getFaceEmbedding(File imageFile) async {
    if (!_isInitialized || _interpreter == null) {
      throw Exception('FaceRecognitionService is not initialized');
    }

    try {
      var bytes = await imageFile.readAsBytes();
      var image = img.decodeImage(bytes);
      if (image == null) throw Exception('Failed to decode image');

      // Apply augmentations and get multiple embeddings
      List<List<double>> augmentedEmbeddings = [];
      for (int i = 0; i < 5; i++) {
        // Generate 5 augmented versions
        var augmentedImage = applyAugmentation(image);
        var resizedImage = img.copyResize(augmentedImage,
            width: INPUT_SIZE, height: INPUT_SIZE);
        var input = _imageToByteListFloat32(resizedImage);
        var outputBuffer = List.generate(
          _interpreter!.getOutputTensor(0).shape[0],
          (_) =>
              List<double>.filled(_interpreter!.getOutputTensor(0).shape[1], 0),
        );
        _interpreter!.run(input, outputBuffer);
        augmentedEmbeddings.add(outputBuffer.expand((list) => list).toList());
      }

      // Average the augmented embeddings
      List<double> averageEmbedding = List.filled(OUTPUT_SIZE, 0);
      for (var embedding in augmentedEmbeddings) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
          averageEmbedding[i] += embedding[i];
        }
      }
      for (int i = 0; i < OUTPUT_SIZE; i++) {
        averageEmbedding[i] /= augmentedEmbeddings.length;
      }

      return averageEmbedding;
    } catch (e) {
      print('Error getting face embedding: $e');
      rethrow;
    }
  }

  img.Image applyAugmentation(img.Image image) {
    int brightnessAdjustment =
        (Random().nextDouble() * 0.6 - 0.3).toInt() * 255;
    img.Image augmentedImage =
        img.adjustColor(image, brightness: brightnessAdjustment);

    if (Random().nextBool()) {
      augmentedImage = img.flipHorizontal(augmentedImage);
    }

    augmentedImage =
        img.copyRotate(augmentedImage, angle: Random().nextDouble() * 20 - 10);
    augmentedImage = img.noise(augmentedImage, 0.1);

    return augmentedImage;
  }

  List<List<List<List<double>>>> _imageToByteListFloat32(img.Image image) {
    var convertedBytes = List.generate(
      1,
      (i) => List.generate(
        INPUT_SIZE,
        (y) => List.generate(
          INPUT_SIZE,
          (x) {
            var pixel = image.getPixel(x, y);
            return [
              (pixel.r.toDouble() - 127.5) / 128,
              (pixel.g.toDouble() - 127.5) / 128,
              (pixel.b.toDouble() - 127.5) / 128,
            ];
          },
        ),
      ),
    );
    return convertedBytes;
  }

  double calculateSimilarity(List<double> embedding1, List<double> embedding2) {
    if (embedding1.length != embedding2.length) {
      print(
          'Warning: Embedding length mismatch: ${embedding1.length} vs ${embedding2.length}');
      var minLength = min(embedding1.length, embedding2.length);
      embedding1 = embedding1.sublist(0, minLength);
      embedding2 = embedding2.sublist(0, minLength);
    }

    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (int i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      norm1 += embedding1[i] * embedding1[i];
      norm2 += embedding2[i] * embedding2[i];
    }

    return dotProduct / (sqrt(norm1) * sqrt(norm2));
  }

  bool isImageQualitySufficient(File imageFile) {
    try {
      img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
      if (image == null) {
        print('Failed to decode image');
        return false;
      }

      if (image.width < 200 || image.height < 200) {
        print('Image resolution too low: ${image.width}x${image.height}');
        return false;
      }

      double averageBrightness = _calculateAverageBrightness(image);
      if (averageBrightness < 0.2 || averageBrightness > 0.8) {
        print('Image brightness out of acceptable range: $averageBrightness');
        return false;
      }

      return true;
    } catch (e) {
      print('Error checking image quality: $e');
      return false;
    }
  }

  double _calculateAverageBrightness(img.Image image) {
    try {
      int totalBrightness = 0;
      int pixelCount = image.width * image.height;

      for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
          img.Pixel pixel = image.getPixel(x, y);
          int r = pixel.r.toInt();
          int g = pixel.g.toInt();
          int b = pixel.b.toInt();
          totalBrightness += ((r + g + b) / 3).round();
        }
      }

      return totalBrightness / (pixelCount * 255);
    } catch (e) {
      print('Error calculating average brightness: $e');
      return 0.0;
    }
  }

  void dispose() {
    _interpreter?.close();
    _faceDetector.close();
    _isInitialized = false;
  }
}
