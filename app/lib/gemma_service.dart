import 'package:flutter_gemma/flutter_gemma.dart';

const String _systemPrompt =
    'You are Holos Medyk, an emergency medical assistant for Ukrainian civilians. '
    'Give clear step-by-step first aid instructions in Ukrainian. '
    'Be concise and direct. Respond only in Ukrainian.';

const String _modelUrl =
    'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm';

class GemmaService {
  dynamic _model;

  Future<void> initialize({
    Function(double)? onProgress,
  }) async {
    await FlutterGemma.installModel(
      modelType: ModelType.gemmaIt,
    ).fromNetwork(
      _modelUrl,
    ).withProgress((progress) {
      onProgress?.call(progress / 100.0);
    }).install();

    _model = await FlutterGemma.getActiveModel(
      maxTokens: 2048,
      preferredBackend: PreferredBackend.gpu,
    );
  }

  Future<String> generate(String userText) async {
    if (_model == null) {
      throw Exception('Model not initialized');
    }

    final chat = await _model.createChat();

    // Add system context then user message
    await chat.addQueryChunk(Message.text(
      text: '$_systemPrompt\n\nUser: $userText',
      isUser: true,
    ));

    final response = await chat.generateChatResponse();
    return response?.token ?? 'Помилка генерації відповіді';
  }

  void dispose() {
    _model?.close();
  }
}
