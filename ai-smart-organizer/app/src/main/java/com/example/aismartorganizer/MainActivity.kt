package com.example.aismartorganizer

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.example.aismartorganizer.databinding.ActivityMainBinding
import com.example.aismartorganizer.ui.assistant.AIAssistantFragment
import com.example.aismartorganizer.ui.insights.InsightsFragment
import com.example.aismartorganizer.ui.knowledgemap.KnowledgeMapFragment
import com.example.aismartorganizer.ui.notes.NotesFragment
import com.example.aismartorganizer.ui.todo.ToDoFragment

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (savedInstanceState == null) switchTab(NotesFragment())

        binding.bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.tab_notes -> switchTab(NotesFragment())
                R.id.tab_map -> switchTab(KnowledgeMapFragment())
                R.id.tab_todo -> switchTab(ToDoFragment())
                R.id.tab_ai -> switchTab(AIAssistantFragment())
                R.id.tab_insights -> switchTab(InsightsFragment())
            }
            true
        }
    }

    private fun switchTab(fragment: Fragment) {
        supportFragmentManager.beginTransaction()
            .setCustomAnimations(android.R.anim.fade_in, android.R.anim.fade_out)
            .replace(R.id.fragmentContainer, fragment)
            .commit()
    }
}
