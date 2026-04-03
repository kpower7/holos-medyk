import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'assistant_state.dart';
import 'home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const HolosMedykApp());
}

class HolosMedykApp extends StatelessWidget {
  const HolosMedykApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AssistantState(),
      child: MaterialApp(
        title: 'Голос Медик',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF1B5E20),
            brightness: Brightness.dark,
          ),
          useMaterial3: true,
        ),
        home: const HomeScreen(),
      ),
    );
  }
}
