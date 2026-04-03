package com.example.aismartorganizer.utils

import com.example.aismartorganizer.data.NoteEntity
import com.example.aismartorganizer.data.TaskEntity

data class InsightResult(
    val totalPasswords: Int,
    val strongPasswords: Int,
    val weakPasswords: Int,
    val duplicatePasswords: Int,
    val mostUsedWord: String,
    val completedTasks: Int,
    val pendingTasks: Int
)

object InsightsAnalyzer {

    private val passwordRegex = Regex("(?<!\\S)(?=.*[A-Za-z])(?=.*\\d)[A-Za-z\\d@#$%^&*!?]{6,}(?!\\S)")

    fun analyze(notes: List<NoteEntity>, tasks: List<TaskEntity>): InsightResult {
        val passwords = mutableListOf<String>()
        val wordFreq = HashMap<String, Int>()

        notes.forEach { note ->
            passwordRegex.findAll(note.content).forEach { passwords.add(it.value) }
            note.content.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.forEach {
                wordFreq[it] = (wordFreq[it] ?: 0) + 1
            }
        }

        val total = passwords.size
        val strong = passwords.count { it.length >= 10 && it.any(Char::isUpperCase) && it.any { c -> !c.isLetterOrDigit() } }
        val weak = total - strong
        val duplicates = passwords.groupingBy { it }.eachCount().count { it.value > 1 }
        val mostUsed = wordFreq.maxByOrNull { it.value }?.key ?: "N/A"

        val completed = tasks.count { it.completed }
        val pending = tasks.size - completed

        return InsightResult(total, strong, weak, duplicates, mostUsed, completed, pending)
    }
}
