# Fonction pour formater les écritures scientifiques
function format_sci(match::SubString)
    # Extraire la partie de la mantisse et de l'exposant à partir de la correspondance
    sci_str = String(match)
    mantisse, exposant = split(sci_str, 'e')
    
    # Garder la partie avant 'e' (mantisse) et après 'e' (exposant)
    exposant_sign = occursin(r"\+|-", exposant) ? exposant[1] : "+"
    exposant_value = exposant[2:end]

    # Retourner le format LaTeX : $mantisse$e$exposant$
    return "\$$mantisse\$e\$$exposant_sign$exposant_value\$"
end

# Fonction pour reformater le contenu du fichier .tex
function reformat_tex_file(input_file::String, output_file::String)
    # Lire le contenu du fichier .tex
    tex_content = read(input_file, String)

    # Expression régulière pour capturer les nombres en écriture scientifique
    regex = r"\b[+-]?\d+\.\d+e[+-]?\d+\b"

    # Appliquer le formatage sur les écritures scientifiques
    formatted_content = replace(tex_content, regex => format_sci)

    # Écrire le contenu formaté dans un nouveau fichier
    open(output_file, "w") do f
        write(f, formatted_content)
    end
end
