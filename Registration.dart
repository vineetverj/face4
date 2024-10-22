import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:fac/database/database_service.dart';
import 'package:fac/facerecongnition.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

class RegistrationScreen extends StatefulWidget {
  const RegistrationScreen({Key? key}) : super(key: key);

  @override
  _RegistrationScreenState createState() => _RegistrationScreenState();
}

class _RegistrationScreenState extends State<RegistrationScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _employeeIdController = TextEditingController();
  final FaceRecognitionService _recognitionService = FaceRecognitionService();
  final DatabaseService _databaseService = DatabaseService();

  late AnimationController _animationController;
  late Animation<double> _animation;
  CameraController? _cameraController;
  List<File> capturedImages = [];
  int currentStep = 0;
  final int totalSteps = 5;
  String _statusMessage = '';
  bool _isProcessing = false;
  bool _isCameraInitialized = false;
  bool _isShowingGuide = false;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _initializeCamera();
    _initializeService();
    _initializeAnimation();
    WidgetsBinding.instance.addPostFrameCallback((_) => _showFaceGuide());
  }

  // Request camera and storage permissions at runtime
  Future<void> _requestPermissions() async {
    if (await Permission.camera.isDenied) {
      await Permission.camera.request();
    }
    if (await Permission.storage.isDenied) {
      await Permission.storage.request();
    }
  }

  // Initialize animation controller for face guide
  void _initializeAnimation() {
    _animationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0, end: 1).animate(_animationController)
      ..addListener(() {
        setState(() {});
      });
    _animationController.repeat(reverse: true);
  }

  // Initialize the camera for front-facing capture
  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.ultraHigh,
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  // Initialize face recognition service
  Future<void> _initializeService() async {
    try {
      await _recognitionService.initialize();
      if (mounted) {
        setState(() {
          _statusMessage = 'Face recognition service initialized';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _statusMessage = 'Error initializing face recognition service: $e';
        });
      }
    }
  }

  // Show the face guide overlay for alignment
  void _showFaceGuide() {
    setState(() {
      _isShowingGuide = true;
    });

    Timer(const Duration(seconds: 5), () {
      if (mounted) {
        setState(() {
          _isShowingGuide = false;
        });
      }
    });
  }

  // Get instructions for each step of face capture
  String _getInstructionForStep(int step) {
    switch (step) {
      case 0:
        return 'Look straight at the camera';
      case 1:
        return 'Turn your head slightly to the left';
      case 2:
        return 'Turn your head slightly to the right';
      case 3:
        return 'Tilt your head up a little';
      case 4:
        return 'Tilt your head down a little';
      default:
        return '';
    }
  }

  // Capture the image and validate it
  Future<void> _captureImage() async {
    if (currentStep >= totalSteps ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      final XFile image = await _cameraController!.takePicture();
      final File imageFile = File(image.path);

      if (await _recognitionService.detectFace(imageFile)) {
        if (_recognitionService.isImageQualitySufficient(imageFile)) {
          // Perform augmentation here
          final augmentedImages = _performAugmentation(imageFile);
          capturedImages.addAll(augmentedImages);
          if (mounted) {
            setState(() {
              currentStep++;
              _statusMessage =
                  'Captured and augmented image ${currentStep} of $totalSteps';
            });
          }

          if (currentStep < totalSteps) {
            _showFaceGuide(); // Show guide for the next step
          } else {
            await _processRegistration(); // Process registration after all steps
          }
        } else {
          setState(() {
            _statusMessage = 'Image quality insufficient. Please try again.';
          });
        }
      } else {
        setState(() {
          _statusMessage = 'No face detected. Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Error capturing image: $e';
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // Perform image augmentation for better recognition
  List<File> _performAugmentation(File originalImageFile) {
    final originalImageBytes = originalImageFile.readAsBytesSync();
    img.Image originalImage = img.decodeImage(originalImageBytes)!;

    List<img.Image> augmentedImages = [];

    // Augmentation 1: Brightness Adjustment
    img.Image brighterImage = img.adjustColor(originalImage, brightness: 0.2);
    img.Image darkerImage = img.adjustColor(originalImage, brightness: -0.2);
    augmentedImages.add(brighterImage);
    augmentedImages.add(darkerImage);

    // Augmentation 2: Contrast Adjustment
    img.Image highContrast = img.adjustColor(originalImage, contrast: 1.5);
    img.Image lowContrast = img.adjustColor(originalImage, contrast: 0.8);
    augmentedImages.add(highContrast);
    augmentedImages.add(lowContrast);

    // Augmentation 3: Horizontal Flip
    img.Image flippedImage = img.flipHorizontal(originalImage);
    augmentedImages.add(flippedImage);

    // Augmentation 4: Rotation
    img.Image rotatedLeft = img.copyRotate(originalImage, angle: -10);
    img.Image rotatedRight = img.copyRotate(originalImage, angle: 10);
    augmentedImages.add(rotatedLeft);
    augmentedImages.add(rotatedRight);

    // Saving augmented images in system temp directory
    List<File> augmentedFiles = [];
    for (var i = 0; i < augmentedImages.length; i++) {
      String tempDir = Directory.systemTemp.path;
      String newPath =
          '$tempDir/${DateTime.now().millisecondsSinceEpoch}_aug_$i.jpg';
      File augmentedFile = File(newPath)
        ..writeAsBytesSync(img.encodeJpg(augmentedImages[i]));
      augmentedFiles.add(augmentedFile);
    }

    return augmentedFiles;
  }

  // Process the registration after capturing images
  Future<void> _processRegistration() async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Processing registration...';
    });

    try {
      List<List<double>> embeddings = [];
      for (var imageFile in capturedImages) {
        List<double> embedding =
            await _recognitionService.getFaceEmbedding(imageFile);
        embeddings.add(embedding);
      }

      // Flatten the list of embeddings to prevent nesting issues
      List<double> averageEmbedding = _calculateAverageEmbedding(embeddings);

      await _databaseService.registerUser(
        employeeId: _employeeIdController.text,
        name: _nameController.text,
        faceEmbeddings: averageEmbedding, // No brackets to avoid nested arrays
      );

      setState(() {
        _statusMessage = 'Registration successful!';
        _nameController.clear();
        _employeeIdController.clear();
        capturedImages.clear();
        currentStep = 0;
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error during registration: $e';
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // Calculate average embeddings for all captured images
  List<double> _calculateAverageEmbedding(List<List<double>> embeddings) {
    int embeddingSize = embeddings[0].length;
    List<double> average = List.filled(embeddingSize, 0.0);

    for (var embedding in embeddings) {
      for (int i = 0; i < embeddingSize; i++) {
        average[i] += embedding[i];
      }
    }

    for (int i = 0; i < embeddingSize; i++) {
      average[i] /= embeddings.length;
    }

    return average;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Registration')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: <Widget>[
            if (_isCameraInitialized && _cameraController != null)
              Stack(
                alignment: Alignment.center,
                children: [
                  OvalCameraPreview(
                    child: CameraPreview(_cameraController!),
                  ),
                  if (_isShowingGuide)
                    AnimatedFaceGuide(step: currentStep, animation: _animation),
                ],
              )
            else
              Container(
                height: 300,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.grey,
                ),
                child: const Center(child: Text('Initializing camera...')),
              ),
            const SizedBox(height: 20),
            Text('Step ${currentStep + 1} of $totalSteps',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            Text(_getInstructionForStep(currentStep),
                style: TextStyle(fontSize: 16)),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed:
                  (_isProcessing || _isShowingGuide) ? null : _captureImage,
              child: Text(_isProcessing ? 'Processing...' : 'Capture Image'),
              style: ElevatedButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                textStyle: const TextStyle(fontSize: 18),
              ),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(labelText: 'Full Name'),
            ),
            TextField(
              controller: _employeeIdController,
              decoration: const InputDecoration(labelText: 'Employee ID'),
            ),
            const SizedBox(height: 20),
            Text(_statusMessage,
                style: const TextStyle(fontSize: 16, color: Colors.blue)),
            if (_isProcessing) const CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    _cameraController?.dispose();
    _nameController.dispose();
    _employeeIdController.dispose();
    super.dispose();
  }
}

// Custom widget for oval camera preview
class OvalCameraPreview extends StatelessWidget {
  final Widget child;

  const OvalCameraPreview({Key? key, required this.child}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ClipOval(
      child: Container(
        width: 300, // You can adjust this size as needed
        height: 400,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.blue, width: 4),
        ),
        child: child,
      ),
    );
  }
}

// Animated guide for face alignment
class AnimatedFaceGuide extends StatelessWidget {
  final int step;
  final Animation<double> animation;

  const AnimatedFaceGuide(
      {Key? key, required this.step, required this.animation})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: FaceGuidePainter(step: step, animationValue: animation.value),
      child: Container(),
    );
  }
}

// Custom painter for face guide
class FaceGuidePainter extends CustomPainter {
  final int step;
  final double animationValue;

  FaceGuidePainter({required this.step, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blue.withOpacity(0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final centerX = size.width / 2;
    final centerY = size.height / 2;
    final radius = size.width * 0.3;

    canvas.save();

    switch (step) {
      case 1:
        canvas.translate(-radius * 0.2 * animationValue, 0);
        break;
      case 2:
        canvas.translate(radius * 0.2 * animationValue, 0);
        break;
      case 3:
        canvas.translate(0, -radius * 0.2 * animationValue);
        break;
      case 4:
        canvas.translate(0, radius * 0.2 * animationValue);
        break;
    }

    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(centerX, centerY),
        width: radius * 1.2,
        height: radius * 1.6,
      ),
      paint,
    );

    final eyeRadius = radius * 0.1;
    final leftEyeCenter =
        Offset(centerX - radius * 0.3, centerY - radius * 0.3);
    final rightEyeCenter =
        Offset(centerX + radius * 0.3, centerY - radius * 0.3);
    canvas.drawCircle(leftEyeCenter, eyeRadius, paint);
    canvas.drawCircle(rightEyeCenter, eyeRadius, paint);

    final mouthRect = Rect.fromCenter(
      center: Offset(centerX, centerY + radius * 0.4),
      width: radius * 0.6,
      height: radius * 0.1,
    );
    canvas.drawArc(mouthRect, 0, 3.14, false, paint);

    final arrowPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    Offset start, end;
    switch (step) {
      case 1:
        start = Offset(centerX + radius * 0.8, centerY);
        end = Offset(centerX + radius * 1.0, centerY);
        break;
      case 2:
        start = Offset(centerX - radius * 0.8, centerY);
        end = Offset(centerX - radius * 1.0, centerY);
        break;
      case 3:
        start = Offset(centerX, centerY + radius * 0.8);
        end = Offset(centerX, centerY + radius * 1.0);
        break;
      case 4:
        start = Offset(centerX, centerY - radius * 0.8);
        end = Offset(centerX, centerY - radius * 1.0);
        break;
      default:
        start = Offset.zero;
        end = Offset.zero;
    }

    canvas.drawLine(start, end, arrowPaint);

    final arrowheadSize = radius * 0.05;
    if (step == 1 || step == 2) {
      canvas.drawLine(
        end,
        Offset(end.dx - arrowheadSize, end.dy - arrowheadSize),
        arrowPaint,
      );
      canvas.drawLine(
        end,
        Offset(end.dx - arrowheadSize, end.dy + arrowheadSize),
        arrowPaint,
      );
    } else if (step == 3 || step == 4) {
      canvas.drawLine(
        end,
        Offset(end.dx - arrowheadSize, end.dy - arrowheadSize),
        arrowPaint,
      );
      canvas.drawLine(
        end,
        Offset(end.dx + arrowheadSize, end.dy - arrowheadSize),
        arrowPaint,
      );
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
