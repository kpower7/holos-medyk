import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'assistant_state.dart';
import 'gemma_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late GemmaService _gemmaService;
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _gemmaService = GemmaService();
    _initModel();
  }

  Future<void> _initModel() async {
    final state = context.read<AssistantState>();
    try {
      debugPrint('[INIT] Starting model initialization...');
      await _gemmaService.initialize(
        onProgress: (progress) {
          debugPrint('[INIT] Progress: ${(progress * 100).toInt()}%');
          state.setModelProgress(progress);
        },
      );
      debugPrint('[INIT] Model ready!');
      state.setModelReady();
    } catch (e, stack) {
      debugPrint('[INIT] ERROR: $e');
      debugPrint('[INIT] STACK: $stack');
      state.setError('Помилка: $e');
    }
  }

  Future<void> _onTapMic() async {
    final state = context.read<AssistantState>();
    if (!state.modelReady) return;
    if (state.phase == AppPhase.speaking) {
      // Interrupt playback
      // TODO: Stop TTS
      state.setPhase(AppPhase.ready);
      return;
    }
    if (state.phase != AppPhase.ready) return;

    // For now: text input dialog as placeholder until audio recording is wired
    final text = await _showTextInput();
    if (text == null || text.isEmpty) return;

    state.addUserMessage(text);
    _scrollToBottom();

    state.setPhase(AppPhase.thinking);
    try {
      final response = await _gemmaService.generate(text);
      state.addAssistantMessage(response);
      _scrollToBottom();

      state.setPhase(AppPhase.speaking);
      // TODO: TTS playback here
      await Future.delayed(const Duration(seconds: 1));
      state.setPhase(AppPhase.ready);
    } catch (e) {
      state.setError('Помилка: $e');
      await Future.delayed(const Duration(seconds: 2));
      state.setPhase(AppPhase.ready);
    }
  }

  Future<String?> _showTextInput() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Опишіть ситуацію'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            hintText: 'Наприклад: моя дочка кровоточить з руки...',
          ),
          maxLines: 3,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Скасувати'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            child: const Text('Надіслати'),
          ),
        ],
      ),
    );
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AssistantState>(
      builder: (context, state, _) {
        return Scaffold(
          body: SafeArea(
            child: Column(
              children: [
                // Header
                _buildHeader(state),

                // Chat messages
                Expanded(
                  child: state.messages.isEmpty
                      ? _buildEmptyState(state)
                      : _buildMessageList(state),
                ),

                // Status bar
                _buildStatusBar(state),

                // Mic button
                _buildMicButton(state),

                const SizedBox(height: 24),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildHeader(AssistantState state) {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Icon(
            Icons.local_hospital,
            color: Theme.of(context).colorScheme.primary,
            size: 32,
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Голос Медик',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'Holos Medyk',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                  ),
                ),
              ],
            ),
          ),
          // Offline indicator
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.green.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.wifi_off, size: 14, color: Colors.green),
                SizedBox(width: 4),
                Text(
                  'OFFLINE',
                  style: TextStyle(fontSize: 11, color: Colors.green),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState(AssistantState state) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.emergency,
              size: 80,
              color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.3),
            ),
            const SizedBox(height: 24),
            Text(
              state.modelReady
                  ? 'Натисніть кнопку та опишіть\nмедичну ситуацію'
                  : 'Підготовка...',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 18, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMessageList(AssistantState state) {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      itemCount: state.messages.length,
      itemBuilder: (context, index) {
        final msg = state.messages[index];
        return _buildMessageBubble(msg);
      },
    );
  }

  Widget _buildMessageBubble(ChatMessage msg) {
    final isUser = msg.isUser;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(14),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.8,
        ),
        decoration: BoxDecoration(
          color: isUser
              ? Theme.of(context).colorScheme.primary
              : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
        ),
        child: Text(
          msg.text,
          style: TextStyle(
            fontSize: 16,
            color: isUser ? Colors.white : null,
          ),
        ),
      ),
    );
  }

  Widget _buildStatusBar(AssistantState state) {
    Color indicatorColor = switch (state.phase) {
      AppPhase.loading  => Colors.orange,
      AppPhase.ready    => Colors.green,
      AppPhase.listening => Colors.red,
      AppPhase.thinking  => Colors.blue,
      AppPhase.speaking  => Colors.purple,
      AppPhase.error     => Colors.red,
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (state.phase == AppPhase.loading)
            SizedBox(
              width: 16, height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                value: state.modelProgress > 0 ? state.modelProgress : null,
              ),
            )
          else
            Container(
              width: 10, height: 10,
              decoration: BoxDecoration(
                color: indicatorColor,
                shape: BoxShape.circle,
              ),
            ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              state.statusText,
              style: const TextStyle(fontSize: 14, color: Colors.grey),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMicButton(AssistantState state) {
    final isActive = state.modelReady &&
        (state.phase == AppPhase.ready || state.phase == AppPhase.speaking);

    final Color buttonColor = switch (state.phase) {
      AppPhase.listening => Colors.red,
      AppPhase.speaking  => Colors.orange,
      _ => Theme.of(context).colorScheme.primary,
    };

    final IconData icon = switch (state.phase) {
      AppPhase.listening => Icons.stop,
      AppPhase.speaking  => Icons.stop,
      _ => Icons.mic,
    };

    return GestureDetector(
      onTap: isActive ? _onTapMic : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 96,
        height: 96,
        decoration: BoxDecoration(
          color: isActive ? buttonColor : Colors.grey.shade800,
          shape: BoxShape.circle,
          boxShadow: isActive
              ? [
                  BoxShadow(
                    color: buttonColor.withValues(alpha: 0.4),
                    blurRadius: 20,
                    spreadRadius: 4,
                  )
                ]
              : null,
        ),
        child: Icon(
          icon,
          size: 48,
          color: isActive ? Colors.white : Colors.grey,
        ),
      ),
    );
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _gemmaService.dispose();
    super.dispose();
  }
}
