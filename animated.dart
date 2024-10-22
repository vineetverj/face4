import 'package:flutter/material.dart';

class AnimatedFaceGuide extends StatefulWidget {
  final int currentStep;

  AnimatedFaceGuide(
      {required this.currentStep,
      required int step,
      required Animation<double> animation,
      required double size});

  @override
  _AnimatedFaceGuideState createState() => _AnimatedFaceGuideState();
}

class _AnimatedFaceGuideState extends State<AnimatedFaceGuide>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller =
        AnimationController(duration: Duration(seconds: 1), vsync: this);
    _animation = Tween<double>(begin: 0, end: 1).animate(_controller);
    _controller.repeat(reverse: true);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return CustomPaint(
          painter: FaceGuidePainter(
            step: widget.currentStep,
            animationValue: _animation.value,
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

class FaceGuidePainter extends CustomPainter {
  final int step;
  final double animationValue;

  FaceGuidePainter({required this.step, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    // Draw different face outlines based on the current step
    // Use animationValue to create movement in the outline
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
