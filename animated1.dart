import 'package:fac/animated.dart';
import 'package:flutter/material.dart';

class AnimatedFaceGuide extends StatelessWidget {
  final int step;
  final Animation<double> animation;
  final double size;

  const AnimatedFaceGuide({
    Key? key,
    required this.step,
    required this.animation,
    this.size = 200.0,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        return CustomPaint(
          size: Size(size, size),
          painter:
              FaceGuidePainter(step: step, animationValue: animation.value),
        );
      },
    );
  }
}
