// lib/screens/resize_image.dart

import 'dart:io';
import 'package:flutter/material.dart';

class ResizeImage extends StatelessWidget {
  final File imageFile;
  final double width;
  final double height;

  const ResizeImage({
    Key? key,
    required this.imageFile,
    this.width = 300,
    this.height = 400,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Image.file(
      imageFile,
      width: width,
      height: height,
      fit: BoxFit.cover,
    );
  }
}
