// lib/screens/home_screen.dart

import 'package:fac/Registration.dart';
import 'package:fac/recognition.dart';
import 'package:flutter/material.dart';
// import 'registration_screen.dart';
// import 'recognition_screen.dart';

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Face Recognition App'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              child: Text('Register Face'),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => RegistrationScreen()),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text('Recognize Face'),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (context) => RealTimeRecognitionScreen()),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
