import 'package:flutter/foundation.dart';

enum AppPhase {
  loading,    // Model loading
  ready,      // Waiting for user to press button
  listening,  // Recording user speech
  thinking,   // LLM generating response
  speaking,   // TTS playing response
  error,      // Something went wrong
}

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

class AssistantState extends ChangeNotifier {
  AppPhase _phase = AppPhase.loading;
  final List<ChatMessage> _messages = [];
  String _statusText = 'Завантаження моделі...';
  bool _modelReady = false;
  double _modelProgress = 0.0;

  AppPhase get phase => _phase;
  List<ChatMessage> get messages => List.unmodifiable(_messages);
  String get statusText => _statusText;
  bool get modelReady => _modelReady;
  double get modelProgress => _modelProgress;

  void setPhase(AppPhase phase) {
    _phase = phase;
    _statusText = switch (phase) {
      AppPhase.loading  => 'Завантаження моделі...',
      AppPhase.ready    => 'Натисніть кнопку та говоріть',
      AppPhase.listening => 'Слухаю...',
      AppPhase.thinking  => 'Думаю...',
      AppPhase.speaking  => 'Говорю...',
      AppPhase.error     => 'Помилка',
    };
    notifyListeners();
  }

  void setModelProgress(double progress) {
    _modelProgress = progress;
    _statusText = 'Завантаження моделі... ${(progress * 100).toInt()}%';
    notifyListeners();
  }

  void setModelReady() {
    _modelReady = true;
    setPhase(AppPhase.ready);
  }

  void setError(String message) {
    _phase = AppPhase.error;
    _statusText = message;
    notifyListeners();
  }

  void addUserMessage(String text) {
    _messages.add(ChatMessage(text: text, isUser: true));
    notifyListeners();
  }

  void addAssistantMessage(String text) {
    _messages.add(ChatMessage(text: text, isUser: false));
    notifyListeners();
  }

  void clearMessages() {
    _messages.clear();
    notifyListeners();
  }
}
