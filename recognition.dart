import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:fac/database/database_service.dart';
import 'package:fac/facerecongnition.dart';

class RealTimeRecognitionScreen extends StatefulWidget {
  @override
  _RealTimeRecognitionScreenState createState() =>
      _RealTimeRecognitionScreenState();
}

class _RealTimeRecognitionScreenState extends State<RealTimeRecognitionScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  final FaceRecognitionService _recognitionService = FaceRecognitionService();
  final DatabaseService _databaseService = DatabaseService();
  String _statusMessage = '';
  bool _isProcessing = false;
  Timer? _recognitionTimer;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeService();
  }

  Future<void> _initializeCamera() async {
    print('Initializing camera...');
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      frontCamera,
      ResolutionPreset.high,
    );

    _initializeControllerFuture = _controller.initialize();
    if (mounted) {
      setState(() {});
      _startRecognitionTimer();
    }
    print('Camera initialized');
  }

  Future<void> _initializeService() async {
    try {
      print('Initializing face recognition service...');
      await _recognitionService.initialize();
      setState(() {
        _statusMessage = 'Face recognition service initialized';
      });
      print('Face recognition service initialized successfully');
    } catch (e) {
      print('Error initializing face recognition service: $e');
      setState(() {
        _statusMessage = 'Error initializing face recognition service: $e';
      });
    }
  }

  void _startRecognitionTimer() {
    print('Starting recognition timer');
    _recognitionTimer = Timer.periodic(Duration(seconds: 3), (_) {
      if (!_isProcessing && _recognitionService.isInitialized) {
        _processFrame();
      }
    });
  }

  Future<void> _processFrame() async {
    if (!_controller.value.isInitialized) {
      print('Camera not initialized');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Processing frame...';
    });

    try {
      print('Taking picture...');
      final image = await _controller.takePicture();
      print('Picture taken, path: ${image.path}');
      final imageFile = File(image.path);

      print('Detecting face...');
      bool hasFace = await _recognitionService.detectFace(imageFile);
      print('Face detected: $hasFace');

      if (!hasFace) {
        setState(() {
          _statusMessage = 'No face detected';
        });
        return;
      }

      print('Generating face embedding...');
      List<double> faceEmbedding =
          await _recognitionService.getFaceEmbedding(imageFile);
      print('Face embedding generated, length: ${faceEmbedding.length}');

      print('Finding matching user...');
      var matchedUser = await _findMatchingUser(faceEmbedding);
      print(
          'Matched user: ${matchedUser != null ? matchedUser['name'] : 'None'}');

      if (matchedUser != null) {
        await _updateAttendance(matchedUser);
      } else {
        setState(() {
          _statusMessage = 'Face not recognized. Please register first.';
        });
      }
    } catch (e) {
      print('Error during recognition: $e');
      setState(() {
        _statusMessage = 'Error during recognition: $e';
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Future<Map<String, dynamic>?> _findMatchingUser(
      List<double> faceEmbedding) async {
    print('Fetching all users from database...');
    var users = await _databaseService.getAllUsers();
    print('Number of registered users: ${users.docs.length}');

    double highestSimilarity = 0;
    Map<String, dynamic>? matchedUser;

    for (var doc in users.docs) {
      List<dynamic> storedEmbedding = doc['embedding'];
      if (storedEmbedding.length != faceEmbedding.length) {
        print('Warning: Embedding length mismatch for user ${doc['name']}');
        continue;
      }
      double similarity = _recognitionService.calculateSimilarity(
        faceEmbedding,
        List<double>.from(storedEmbedding),
      );
      print('Similarity with user ${doc['name']}: $similarity');

      if (similarity > highestSimilarity &&
          similarity >= FaceRecognitionService.THRESHOLD) {
        highestSimilarity = similarity;
        matchedUser = doc.data() as Map<String, dynamic>;
        matchedUser['id'] = doc.id;
      }
    }

    return matchedUser;
  }

  Future<void> _updateAttendance(Map<String, dynamic> user) async {
    print('Updating attendance for user: ${user['name']}');
    bool currentAttendance = user['attendance'] ?? false;
    await _databaseService.updateAttendance(
      employeeId: user['id'],
      isCheckIn: !currentAttendance,
    );

    setState(() {
      _statusMessage = currentAttendance
          ? 'Goodbye, ${user['name']}! Check-out successful.'
          : 'Welcome, ${user['name']}! Check-in successful.';
    });
    print('Attendance updated: $_statusMessage');
  }

  @override
  void dispose() {
    print('Disposing RealTimeRecognitionScreen');
    _recognitionTimer?.cancel();
    _controller.dispose();
    _recognitionService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Real-Time Face Recognition')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Column(
              children: [
                Expanded(
                  child: CameraPreview(_controller),
                ),
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(_statusMessage, textAlign: TextAlign.center),
                ),
                if (_isProcessing) CircularProgressIndicator(),
              ],
            );
          } else {
            return Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}
