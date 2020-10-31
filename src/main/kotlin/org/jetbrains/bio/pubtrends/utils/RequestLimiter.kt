package org.jetbrains.bio.pubtrends.utils

import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

class RequestLimiter(maxR_ : Int, interval_ : Long) {
    var maxRequests : AtomicInteger = AtomicInteger(0)
    var interval : AtomicLong = AtomicLong(0)
    init {
        maxRequests.set(maxR_)
        interval.set(interval_)
    }
    private val queue = ConcurrentLinkedQueue<LocalDateTime>()
    private val queueSize = AtomicInteger(0)
    //set how many requests we can make in `interval` milliseconds
    fun set(newMaxRequests : Int, newInterval : Long) {
        maxRequests.set(newMaxRequests)
        interval.set(newInterval)
    }

    @Synchronized fun waitForRequest() {
        if (maxRequests.get() != 0) {
            while (queueSize.getAcquire() >= maxRequests.getAcquire()) {
                if (getDif(queue.peek(), LocalDateTime.now()) > interval.getAcquire()) {
                    queue.remove()
                    queueSize.decrementAndGet()
                }
            }
        }
        queue.add(LocalDateTime.now())
        queueSize.incrementAndGet()
    }

    private fun getDif(time1: LocalDateTime, time2: LocalDateTime) : Long {
        return ChronoUnit.MILLIS.between(time1, time2)
    }
}