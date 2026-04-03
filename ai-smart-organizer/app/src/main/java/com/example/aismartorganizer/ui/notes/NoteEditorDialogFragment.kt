package com.example.aismartorganizer.ui.notes

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import androidx.fragment.app.activityViewModels
import com.example.aismartorganizer.R
import com.example.aismartorganizer.SmartOrganizerApp
import com.example.aismartorganizer.databinding.FragmentNoteEditorBinding
import com.example.aismartorganizer.viewmodel.MainViewModel
import com.example.aismartorganizer.viewmodel.ViewModelFactory

class NoteEditorDialogFragment : BottomSheetDialogFragment() {

    private var _binding: FragmentNoteEditorBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MainViewModel by activityViewModels {
        ViewModelFactory((requireActivity().application as SmartOrganizerApp).repository)
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentNoteEditorBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.btnSave.setOnClickListener {
            val title = binding.inputTitle.text?.toString().orEmpty().ifBlank { "Untitled" }
            val content = binding.inputContent.text?.toString().orEmpty()
            viewModel.addNote(title, content, R.color.accentYellow)
            dismiss()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
