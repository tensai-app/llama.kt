package com.tensai.llamakt

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PerfTimingsTest {

    @Test
    fun readsTheFourNativeFiguresInOrder() {
        val perf = PerfTimings.of(doubleArrayOf(2088.0, 67_660.0, 127.0, 24_380.0))!!
        assertEquals(PerfTimings(promptTokens = 2088, promptMs = 67_660.0, evalTokens = 127, evalMs = 24_380.0), perf)
        assertEquals(30.86, perf.promptTokPerSec!!, 0.01)
        assertEquals(5.21, perf.evalTokPerSec!!, 0.01)
    }

    @Test
    fun aCountWithNoTimeIsNoFigure() {
        // llama.cpp reports n_eval as at least 1, even when nothing was decoded.
        val perf = PerfTimings.of(doubleArrayOf(31.0, 770.0, 1.0, 0.0))!!
        assertEquals(0, perf.evalTokens)
        assertNull(perf.evalTokPerSec)
        assertEquals(40.26, perf.promptTokPerSec!!, 0.01)
    }

    @Test
    fun anythingButFourValuesIsNull() {
        assertNull(PerfTimings.of(DoubleArray(0)))
        assertNull(PerfTimings.of(doubleArrayOf(1.0, 2.0, 3.0)))
    }
}
