package com.tensai.llamakt

import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch

/**
 * Stream completion tokens as a [Flow].
 *
 * Usage:
 *   engine.load("/sdcard/model.gguf")
 *   engine.decode("Hello").collect { token -> print(token) }
 *
 * [params] controls generation length and sampling — see [SamplingParams].
 * The nPredict cap is a hard stop against infinite generation.
 *
 * The flow runs on [Dispatchers.Default]. Cancelling the collector calls
 * [LlamaEngine.interrupt], which ends the native decode early; the flow
 * completes once it has returned.
 */
fun LlamaEngine.decode(
    prompt: String,
    params: SamplingParams = SamplingParams(),
): Flow<String> = interruptibleCompletion(
    resetInterrupt = ::resetInterrupt,
    interrupt = ::interrupt,
) { callback -> completion(prompt, params, callback) }

/**
 * [decode]'s body over the three engine calls it makes, so it can be tested without the native
 * library. The blocking [completion] runs in its own coroutine, so the flow reaches `awaitClose`
 * at once and a cancel interrupts the decode in flight: run inline, it held the flow until the
 * decode had ended on its own. [resetInterrupt] clears what the previous decode left, before the
 * last cancellation check, so a cancel landing after that check is never erased. [interrupt] only
 * fires on a decode still running: one fired after a normal end would stay pending, the interrupt
 * being sticky.
 */
internal fun interruptibleCompletion(
    resetInterrupt: () -> Unit,
    interrupt: () -> Unit,
    completion: (TokenCallback) -> Int,
): Flow<String> = callbackFlow {
    val callback = TokenCallback { token -> trySend(token) }
    val returned = AtomicBoolean(false)
    launch {
        resetInterrupt()
        ensureActive()
        completion(callback)
        returned.set(true)
        close()
    }
    awaitClose { if (!returned.get()) interrupt() }
}.flowOn(Dispatchers.Default)

/**
 * Chat-mode Flow: applies the model's built-in chat template to [messages],
 * then streams the response tokens.
 *
 * The template is read from the GGUF metadata — works with any model
 * (Qwen, Llama, Mistral, …) without hardcoding a format. EOS is emitted
 * naturally by the template, and [SamplingParams.nPredict] acts as a hard cap.
 *
 * Usage:
 *   val messages = listOf(ChatMessage("user", "Hello, who are you?"))
 *   engine.chat(messages).collect { token -> print(token) }
 */
fun LlamaEngine.chat(
    messages: List<ChatMessage>,
    params: SamplingParams = SamplingParams(),
    enableThinking: Boolean = true,
): Flow<String> {
    val prompt = formatChat(messages, enableThinking)
    return decode(prompt, params)
}
