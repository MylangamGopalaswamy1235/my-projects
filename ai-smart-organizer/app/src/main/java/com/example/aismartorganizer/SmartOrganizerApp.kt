package com.example.aismartorganizer

import android.app.Application
import com.example.aismartorganizer.data.AppDatabase
import com.example.aismartorganizer.data.Repository

class SmartOrganizerApp : Application() {
    lateinit var repository: Repository

    override fun onCreate() {
        super.onCreate()
        val db = AppDatabase.getInstance(this)
        repository = Repository(db.noteDao(), db.taskDao())
    }
}
