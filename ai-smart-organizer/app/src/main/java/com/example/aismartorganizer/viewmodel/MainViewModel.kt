package com.example.aismartorganizer.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.aismartorganizer.data.NoteEntity
import com.example.aismartorganizer.data.Repository
import com.example.aismartorganizer.data.TaskEntity
import kotlinx.coroutines.launch
import java.util.Stack

class MainViewModel(private val repository: Repository) : ViewModel() {
    val notes: LiveData<List<NoteEntity>> = repository.notes
    val tasks: LiveData<List<TaskEntity>> = repository.tasks

    // Stack DS for undoing latest note insert.
    private val noteStack = Stack<NoteEntity>()

    fun addNote(title: String, content: String, colorTag: Int, parentId: Int? = null) {
        val note = NoteEntity(
            title = title,
            content = content,
            timestamp = System.currentTimeMillis(),
            colorTag = colorTag,
            parentId = parentId
        )
        noteStack.push(note)
        viewModelScope.launch { repository.addNote(note) }
    }

    fun undoLastNote() {
        if (noteStack.isNotEmpty()) {
            // Basic stack pop: in this beginner demo we only show stack usage state-wise.
            noteStack.pop()
        }
    }

    fun addTask(title: String, priority: String) {
        viewModelScope.launch {
            repository.addTask(TaskEntity(title = title, priority = priority))
        }
    }

    fun toggleTask(task: TaskEntity, checked: Boolean) {
        viewModelScope.launch { repository.updateTask(task.copy(completed = checked)) }
    }
}
