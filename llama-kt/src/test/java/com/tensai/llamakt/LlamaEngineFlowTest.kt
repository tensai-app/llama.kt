package com.tensai.llamakt

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** [interruptibleCompletion], [decode]'s body, against a fake engine with the sticky interrupt. */
class LlamaEngineFlowTest {

    private class FakeEngine(private val nPredict: Int) {
        val interrupted = AtomicBoolean(false)
        val interrupts = AtomicInteger(0)
        val resets = AtomicInteger(0)

        fun interrupt() {
            interrupts.incrementAndGet()
            interrupted.set(true)
        }

        fun resetInterrupt() {
            resets.incrementAndGet()
            interrupted.set(false)
        }

        /** One token per step until interrupted or [nPredict]; bounded so a regression is a red test. */
        fun completion(callback: TokenCallback): Int {
            val deadline = System.nanoTime() + 3_000_000_000L
            var n = 0
            while (n < nPredict && !interrupted.get() && System.nanoTime() < deadline) {
                callback.onToken("t$n")
                n++
                Thread.sleep(1)
            }
            return n
        }

        fun flow() = interruptibleCompletion(::resetInterrupt, ::interrupt, ::completion)
    }

    @Test
    fun cancellingTheCollectorInterruptsTheDecodeInFlight() = runBlocking {
        val engine = FakeEngine(nPredict = 1_000_000)

        // first() cancels the flow after one token; the flow must still complete.
        withTimeout(2_000) { engine.flow().first() }

        assertEquals("the cancel reached the native decode", 1, engine.interrupts.get())
        assertTrue(engine.interrupted.get())
    }

    @Test
    fun aDecodeThatEndsOnItsOwnLeavesNoInterruptPending() = runBlocking {
        val engine = FakeEngine(nPredict = 5)

        val tokens = withTimeout(2_000) { engine.flow().toList() }

        assertEquals(5, tokens.size)
        assertEquals(0, engine.interrupts.get())
        assertFalse("a pending interrupt would stop the next decode", engine.interrupted.get())
    }

    @Test
    fun anInterruptLeftBeforeTheFlowIsClearedFirst() = runBlocking {
        val engine = FakeEngine(nPredict = 5)
        engine.interrupt()

        val tokens = withTimeout(2_000) { engine.flow().toList() }

        assertEquals(1, engine.resets.get())
        assertEquals(5, tokens.size)
    }
}
