package com.example.aismartorganizer.ui.notes

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aismartorganizer.R
import com.example.aismartorganizer.SmartOrganizerApp
import com.example.aismartorganizer.adapter.NotesAdapter
import com.example.aismartorganizer.databinding.FragmentNotesBinding
import com.example.aismartorganizer.utils.DataStructureUtils
import com.example.aismartorganizer.viewmodel.MainViewModel
import com.example.aismartorganizer.viewmodel.ViewModelFactory

class NotesFragment : Fragment() {
    private var _binding: FragmentNotesBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MainViewModel by activityViewModels {
        ViewModelFactory((requireActivity().application as SmartOrganizerApp).repository)
    }

    private val adapter = NotesAdapter { note ->
        val all = viewModel.notes.value.orEmpty()
        val relatedGraph = DataStructureUtils.buildRelatedGraph(all)
        val relatedIds = relatedGraph[note.id].orEmpty()
        val relatedTitles = all.filter { it.id in relatedIds }.joinToString { it.title }
        binding.relatedNotes.text = if (relatedTitles.isBlank()) "No related notes" else relatedTitles
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentNotesBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.notesRecycler.layoutManager = LinearLayoutManager(requireContext())
        binding.notesRecycler.adapter = adapter

        viewModel.notes.observe(viewLifecycleOwner) { adapter.submitList(it) }

        binding.fabAddNote.setOnClickListener {
            NoteEditorDialogFragment().show(parentFragmentManager, "editor")
        }

        binding.btnUndo.setOnClickListener { viewModel.undoLastNote() }

        seedDemoIfEmpty()
    }

    private fun seedDemoIfEmpty() {
        if (viewModel.notes.value.isNullOrEmpty()) {
            viewModel.addNote("DBMS Basics", "password DBMS123! normalization keys", R.color.accentYellow)
            viewModel.addNote("DBMS Revision", "Revise joins and transactions", R.color.lowPriority)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
