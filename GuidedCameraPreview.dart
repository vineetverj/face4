// guided_camera_preview.dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class GuidedCameraPreview extends StatelessWidget {
  final CameraController controller;

  const GuidedCameraPreview({Key? key, required this.controller})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return Container();
    }
    return AspectRatio(
      aspectRatio: controller.value.aspectRatio,
      child: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller),
          CustomPaint(
            painter: FaceOverlayPainter(),
          ),
        ],
      ),
    );
  }
}

class FaceOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final ovalRect = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2),
      width: size.width * 0.7,
      height: size.height * 0.8,
    );

    canvas.drawOval(ovalRect, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
