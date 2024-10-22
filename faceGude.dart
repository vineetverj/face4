import 'package:flutter/material.dart';

class FaceGuidePainter extends CustomPainter {
  final int step;
  final double animationValue;

  FaceGuidePainter({required this.step, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final centerX = size.width / 2;
    final centerY = size.height / 2;
    final radius = size.width * 0.3;

    canvas.save();

    // Apply transformations based on step
    switch (step) {
      case 0: // Straight
        // No transformation needed
        break;
      case 1: // Left
        canvas.rotate(-0.2 * animationValue);
        canvas.translate(-radius * 0.2 * animationValue, 0);
        break;
      case 2: // Right
        canvas.rotate(0.2 * animationValue);
        canvas.translate(radius * 0.2 * animationValue, 0);
        break;
      case 3: // Up
        canvas.rotate(-0.1 * animationValue);
        canvas.translate(0, -radius * 0.2 * animationValue);
        break;
      case 4: // Down
        canvas.rotate(0.1 * animationValue);
        canvas.translate(0, radius * 0.2 * animationValue);
        break;
    }

    // Draw face outline
    canvas.drawOval(
      Rect.fromCenter(
          center: Offset(centerX, centerY),
          width: radius * 2,
          height: radius * 2.6),
      paint,
    );

    // Draw eyes
    final eyeRadius = radius * 0.15;
    canvas.drawCircle(Offset(centerX - radius * 0.4, centerY - radius * 0.2),
        eyeRadius, paint);
    canvas.drawCircle(Offset(centerX + radius * 0.4, centerY - radius * 0.2),
        eyeRadius, paint);

    // Draw mouth
    final mouthRect = Rect.fromCenter(
      center: Offset(centerX, centerY + radius * 0.4),
      width: radius * 0.8,
      height: radius * 0.2,
    );
    canvas.drawArc(mouthRect, 0, 3.14, false, paint);

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
