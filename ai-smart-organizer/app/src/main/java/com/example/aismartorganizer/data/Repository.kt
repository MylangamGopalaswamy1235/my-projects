package com.example.aismartorganizer.data

class Repository(private val noteDao: NoteDao, private val taskDao: TaskDao) {
    val notes = noteDao.getAllNotes()
    val tasks = taskDao.getAllTasks()

    suspend fun addNote(note: NoteEntity) = noteDao.insert(note)
    suspend fun addTask(task: TaskEntity) = taskDao.insert(task)
    suspend fun updateTask(task: TaskEntity) = taskDao.update(task)
}
